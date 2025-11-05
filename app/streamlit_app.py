import os, io, json, base64, zipfile, requests
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from utils import parse_line, to_minute_features, FEATURES

ART_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SCALER_PATH = os.path.join(ART_DIR, "scaler.joblib")
ISO_PATH    = os.path.join(ART_DIR, "isoforest.joblib")
SPEC_PATH   = os.path.join(ART_DIR, "feature_spec.json")
CTX_CSV     = os.path.join(ART_DIR, "all_minutes_with_scores.csv")  # optional

# Load artifacts
scaler = load(SCALER_PATH)
iso    = load(ISO_PATH)
spec   = json.load(open(SPEC_PATH))
FINAL_FEATURES = spec["features"]
rules = spec["post_rules"]
ROLL  = spec.get("rolling", {}).get("window_minutes", 60)

context_df = None
if os.path.exists(CTX_CSV):
    context_df = pd.read_csv(CTX_CSV, parse_dates=["minute"])

st.title("NASA Log Anomaly Detector")

push_to_repo = st.checkbox("Also commit uploaded logs to GitHub (triggers retraining)")
uploaded = st.file_uploader("Upload access log (.log/.txt/.zip)", type=["log","txt","zip"])

def read_text_stream(name: str, raw: bytes):
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            # pick largest file that looks like a log
            cands = [zi for zi in zf.infolist() if ("access" in zi.filename.lower() or zi.filename.lower().endswith((".log",".txt")))]
            target = max(cands, key=lambda z: z.file_size) if cands else zf.infolist()[0]
            return io.TextIOWrapper(zf.open(target), errors="ignore")
    return io.TextIOWrapper(io.BytesIO(raw), errors="ignore")

def build_context(df_sample, hours_context=3):
    if context_df is None or context_df.empty:
        return df_sample[["minute","req_count","error_rate"]].copy()
    t0 = df_sample["minute"].min()
    mask = (context_df["minute"] < t0) & (context_df["minute"] >= (t0 - pd.Timedelta(hours=hours_context)))
    ctx = context_df.loc[mask, ["minute","req_count","error_rate"]].copy()
    if len(ctx) < 5:
        ctx = context_df[["minute","req_count","error_rate"]].tail(hours_context*60).copy()
    return ctx

def add_rolling(df_core, ctx):
    eps = 1e-9
    comb = pd.concat([ctx, df_core[["minute","req_count","error_rate"]]], ignore_index=True).sort_values("minute")
    comb["req_roll_mean"] = comb["req_count"].rolling(ROLL, min_periods=5).mean()
    comb["req_roll_std"]  = comb["req_count"].rolling(ROLL, min_periods=5).std()
    comb["err_roll_mean"] = comb["error_rate"].rolling(ROLL, min_periods=5).mean()
    comb["err_roll_std"]  = comb["error_rate"].rolling(ROLL, min_periods=5).std()
    comb["req_count_ratio"]  = comb["req_count"] / (comb["req_roll_mean"] + eps)
    comb["error_rate_ratio"] = comb["error_rate"] / (comb["err_roll_mean"] + eps)
    comb["req_z"] = (comb["req_count"] - comb["req_roll_mean"]) / (comb["req_roll_std"] + eps)
    comb["err_z"] = (comb["error_rate"] - comb["err_roll_mean"]) / (comb["err_roll_std"] + eps)
    return comb

def score(feat_full: pd.DataFrame):
    Xs = scaler.transform(feat_full[FINAL_FEATURES].values.astype(float))
    pred  = iso.predict(Xs)
    score = iso.score_samples(Xs)
    feat_full["iso_score"]  = score
    feat_full["is_anomaly"] = (pred == -1).astype(int)

    MIN_TRAFFIC      = rules["min_traffic"]
    HIGH_ERR         = rules["high_error_threshold"]
    FIVEXX_PRIORITY  = rules["five_xx_priority"]

    rule_model        = (feat_full["is_anomaly"] == 1)
    rule_high_err     = (feat_full["req_count"] >= MIN_TRAFFIC) & (feat_full["error_rate"] >= HIGH_ERR)
    rule_5xx_priority = (feat_full["status_5xx"] >= FIVEXX_PRIORITY)
    feat_full["is_actionable"] = (rule_model & (feat_full["req_count"] >= MIN_TRAFFIC)) | rule_high_err | rule_5xx_priority

    return feat_full

# Optional GitHub commit helper (using dotenv-loaded env)
GH_OWNER  = os.getenv("GH_OWNER")
GH_REPO   = os.getenv("GH_REPO")
GH_BRANCH = os.getenv("GH_BRANCH", "main")
GH_TOKEN  = os.getenv("GH_TOKEN")

def commit_to_github(path_in_repo: str, content_bytes: bytes, message: str):
    assert GH_OWNER and GH_REPO and GH_TOKEN, "GitHub env vars missing."
    url = f"https://api.github.com/repos/{GH_OWNER}/{GH_REPO}/contents/{path_in_repo}"
    headers = {"Authorization": f"token {GH_TOKEN}", "Accept": "application/vnd.github+json"}
    # Read existing to get sha if present
    sha = None
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        sha = r.json().get("sha")

    data = {
        "message": message,
        "branch": GH_BRANCH,
        "content": base64.b64encode(content_bytes).decode("utf-8")
    }
    if sha:
        data["sha"] = sha

    resp = requests.put(url, headers=headers, data=json.dumps(data))
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"GitHub commit failed: {resp.status_code} {resp.text}")
    return resp.json()["content"]["path"]

if uploaded:
    raw_bytes = uploaded.read()
    text_io = read_text_stream(uploaded.name, raw_bytes)

    # Parse
    parsed = []
    for line in text_io:
        r = parse_line(line)
        if r: parsed.append(r)

    df = to_minute_features(parsed)
    if df.empty:
        st.error("No valid lines parsed from file."); st.stop()

    # Rolling features via context
    ctx = build_context(df)
    comb = add_rolling(df, ctx)
    feat_full = df.merge(
        comb[["minute","req_count_ratio","error_rate_ratio","req_z","err_z"]],
        on="minute", how="left"
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Score
    feat_scored = score(feat_full).sort_values("minute")
    # === SIMPLE TEXT VERDICT + COLOR BADGE ===

    # thresholds (tunable)
    ERR_RATE_WARN = 0.05      # 5% starts to be mildly concerning
    Z_ERR_STRONG  = 2.0       # err_z > 2σ -> unusual
    Z_REQ_STRONG  = 3.0       # traffic spike/dip beyond 3σ -> unusual

    def classify_minute(row):
        # Highest priority: explicit actionability or heavy 5xx/error conditions
        if bool(row.get("is_actionable", False)):
            return "definitely anomaly", "#d9342b", 3  # red
        if int(row.get("is_anomaly", 0)) == 1:
            return "very unusual", "#f08c00", 2        # orange
        if (float(row.get("error_rate", 0.0)) >= ERR_RATE_WARN) or \
        (abs(float(row.get("err_z", 0.0))) >= Z_ERR_STRONG) or \
        (abs(float(row.get("req_z", 0.0))) >= Z_REQ_STRONG):
            return "slightly unusual", "#ffd43b", 1    # yellow
        return "normal", "#2fa84f", 0                  # green

    # Ensure we have the needed z-features; if not present in your table view, they still
    # exist in feat_scored from earlier steps. (If you renamed, adjust keys here.)
    if "req_z" not in feat_scored.columns:
        feat_scored["req_z"] = 0.0
    if "err_z" not in feat_scored.columns:
        feat_scored["err_z"] = 0.0

    verdicts = feat_scored.apply(classify_minute, axis=1, result_type="expand")
    feat_scored["verdict"] = verdicts[0]
    feat_scored["_color"]  = verdicts[1]
    feat_scored["_sev"]    = verdicts[2]

    # Overall verdict = worst severity across minutes
    sev_to_label = {
        0: ("normal", "#2fa84f"),
        1: ("slightly unusual", "#ffd43b"),
        2: ("very unusual", "#f08c00"),
        3: ("definitely anomaly", "#d9342b"),
    }
    overall_sev = int(feat_scored["_sev"].max()) if len(feat_scored) else 0
    overall_label, overall_color = sev_to_label[overall_sev]

    # Pretty badge
    def badge(text, color):
        return f"""
        <div style="
        display:inline-block;
        padding:8px 12px;
        border-radius:999px;
        background:{color};
        color:#0b0c0d;
        font-weight:600;">
        {text.upper()}
        </div>"""

    st.markdown("### Verdict")
    st.markdown(badge(overall_label, overall_color), unsafe_allow_html=True)

    # Optional: small counts per class
    counts = (
        feat_scored["verdict"]
        .value_counts()
        .reindex(["normal","slightly unusual","very unusual","definitely anomaly"], fill_value=0)
    )
    # st.write(
    #     {
    #         "normal": int(counts["normal"]),
    #         "slightly unusual": int(counts["slightly unusual"]),
    #         "very unusual": int(counts["very unusual"]),
    #         "definitely anomaly": int(counts["definitely anomaly"]),
    #     }
    # )
    st.divider()   # or: st.markdown("---")


    # Show table with verdict (color shown via text; Streamlit's dataframe doesn't support row color styling)
    show_cols = ["minute","req_count","status_4xx","status_5xx","error_rate",
                "iso_score","is_anomaly","is_actionable","verdict"]
    st.dataframe(feat_scored[show_cols].sort_values("minute"))

    # Keep download button below if you had it:
    st.download_button(
        "Download results CSV",
        feat_scored.drop(columns=["_color","_sev"]).to_csv(index=False),
        file_name="anomaly_results.csv",
        mime="text/csv"
    )


    # Optional: push to GitHub (to trigger retraining workflow)
    if push_to_repo:
        # always commit as zip
        if not uploaded.name.lower().endswith(".zip"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("access.log", raw_bytes)
            payload = buf.getvalue()
            commit_name = "access.zip"
        else:
            payload = raw_bytes
            commit_name = uploaded.name

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        repo_path = f"data/raw/{ts}_{commit_name}"
        try:
            p = commit_to_github(repo_path, payload, f"chore: add uploaded log {commit_name}")
            st.info(f"Pushed to GitHub: `{p}` — retraining workflow will start on push.")
        except Exception as e:
            st.error(f"GitHub commit failed: {e}")
