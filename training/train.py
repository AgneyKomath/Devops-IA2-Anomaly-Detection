# training/train.py
import os, json, glob, math, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ===== import your existing helpers =====
# If utils.py is not importable here, paste parse_line/to_minute_features/FEATURES.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app"))
from utils import parse_line, to_minute_features, FEATURES  # noqa

# ===== knobs via env =====
ROLL_MINUTES      = int(os.getenv("ROLLING_WINDOW_MINUTES", "60"))
MIN_PERIODS       = int(os.getenv("ROLL_MIN_PERIODS", "30"))
CONTAM            = float(os.getenv("CONTAMINATION", "0.02"))
RANDOM_STATE      = int(os.getenv("RANDOM_STATE", "42"))
TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", "45"))
ARCHIVE_FRAC      = float(os.getenv("ARCHIVE_FRAC", "0.1"))  # 10% from older archive, optional

DATA_DIR = "data/raw"
OUT_DIR  = "models"
os.makedirs(OUT_DIR, exist_ok=True)

EXTRA_FEATURES = ["req_count_ratio","error_rate_ratio","req_z","err_z"]
FINAL_FEATURES = FEATURES + EXTRA_FEATURES

def _read_all_lines():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*"), recursive=True))
    for path in files:
        if os.path.isdir(path): 
            continue
        # handle zip or text
        if path.lower().endswith(".zip"):
            import zipfile
            with zipfile.ZipFile(path) as zf:
                # pick likely log file(s)
                members = [zi for zi in zf.infolist() if ("access" in zi.filename.lower() or zi.filename.lower().endswith((".log",".txt")))]
                if not members: 
                    members = zf.infolist()
                for m in members:
                    with zf.open(m) as fh:
                        for raw in fh:
                            yield raw.decode("utf-8", errors="ignore")
        else:
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    yield line

def load_parsed_rows():
    rows = []
    for line in _read_all_lines():
        # print(line);
        r = parse_line(line)
        if r: rows.append(r)
    return rows

def add_rolling(df):
    eps = 1e-9
    df = df.sort_values("minute").reset_index(drop=True).copy()
    df["req_roll_mean"] = df["req_count"].rolling(ROLL_MINUTES, min_periods=MIN_PERIODS).mean()
    df["req_roll_std"]  = df["req_count"].rolling(ROLL_MINUTES, min_periods=MIN_PERIODS).std()
    df["err_roll_mean"] = df["error_rate"].rolling(ROLL_MINUTES, min_periods=MIN_PERIODS).mean()
    df["err_roll_std"]  = df["error_rate"].rolling(ROLL_MINUTES, min_periods=MIN_PERIODS).std()
    df["req_count_ratio"]  = df["req_count"] / (df["req_roll_mean"] + eps)
    df["error_rate_ratio"] = df["error_rate"] / (df["err_roll_mean"] + eps)
    df["req_z"] = (df["req_count"] - df["req_roll_mean"]) / (df["req_roll_std"] + eps)
    df["err_z"] = (df["error_rate"] - df["err_roll_mean"]) / (df["err_roll_std"] + eps)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def main():
    rows = load_parsed_rows()
    if not rows:
        raise SystemExit("No logs found under data/raw/")

    # Base minute features for all rows we have
    base = to_minute_features(rows)

    # --- Sliding window selection (simple & robust) ---

    if base.empty:
        raise SystemExit("No parsed minutes found under data/raw/**.")

    data_end = pd.to_datetime(base["minute"].max())
    anchor   = data_end
    cutoff   = anchor - timedelta(days=TRAIN_WINDOW_DAYS)

    recent   = base[base["minute"] >= cutoff].copy()
    archive  = base[base["minute"] < cutoff].copy()

    # If nothing recent (e.g., historical dataset), pull a minimum chunk from archive
    if recent.empty and not archive.empty:
        min_keep = min(len(archive), max(ROLL_MINUTES * 2, 200))  # ≥ ~2 hours or 200 rows
        recent   = archive.tail(min_keep).copy()
        archive  = archive.iloc[:-min_keep].copy()

    # Optional archive sampling; ensure at least 1 row if sampling applies
    if ARCHIVE_FRAC > 0 and not archive.empty:
        n_samp  = max(int(np.ceil(len(archive) * ARCHIVE_FRAC)), 1)
        archive = archive.sample(n=n_samp, random_state=RANDOM_STATE, replace=False)

    df_train = pd.concat([recent, archive], ignore_index=True).sort_values("minute")
    if df_train.empty:
        raise SystemExit("No trainable minutes selected. Increase TRAIN_WINDOW_DAYS or set USE_REAL_NOW=0.")

    # Add rolling context inside the selected window
    df_ctx = add_rolling(df_train)
    # Train
    X = df_ctx[FINAL_FEATURES].values.astype(float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    iso = IsolationForest(
        n_estimators=300,
        contamination=CONTAM,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ).fit(Xs)

    # Save artifacts
    from joblib import dump
    dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
    dump(iso,    os.path.join(OUT_DIR, "isoforest.joblib"))

    spec = {
        "features": FINAL_FEATURES,
        "post_rules": {"min_traffic": 10, "high_error_threshold": 0.20, "five_xx_priority": 3},
        "rolling": {"window_minutes": ROLL_MINUTES, "min_periods": MIN_PERIODS}
    }
    with open(os.path.join(OUT_DIR, "feature_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)

    # Optional report (handy for the Streamlit rolling context)
    df_ctx["iso_score"]  = iso.score_samples(scaler.transform(df_ctx[FINAL_FEATURES].values.astype(float)))
    df_ctx["is_anomaly"] = (iso.predict(scaler.transform(df_ctx[FINAL_FEATURES].values.astype(float))) == -1).astype(int)
    df_ctx.to_csv(os.path.join(OUT_DIR, "all_minutes_with_scores.csv"), index=False)

    # Sanity prints (visible in CI logs)
    print("✅ retrain complete")
    print(f"minutes_used={len(df_ctx)}  anomaly_rate={(df_ctx['is_anomaly'].mean() if len(df_ctx) else 0):.3f}")

if __name__ == "__main__":
    main()
