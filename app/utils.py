import re, math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from dateutil import tz
import numpy as np
import pandas as pd

LOG_RE = re.compile(
    r'^(?P<host>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] "(?P<method>\S+)\s(?P<path>[^"]*)\s(?P<proto>[^"]+)" (?P<status>\d{3}) (?P<bytes>\S+)'
)

# e.g. [01/Aug/1995:00:00:01 -0400]
def parse_ts(s: str) -> datetime:
    # day/Mon/year:HH:MM:SS ±HHMM
    dt = datetime.strptime(s, "%d/%b/%Y:%H:%M:%S %z")
    return dt.astimezone(timezone.utc)

def parse_line(line: str):
    m = LOG_RE.match(line.strip())
    if not m:
        return None
    d = m.groupdict()
    try:
        ts = parse_ts(d["ts"])
    except Exception:
        return None
    try:
        status = int(d["status"])
    except:
        status = 0
    try:
        byt = 0 if d["bytes"] == "-" else int(d["bytes"])
    except:
        byt = 0
    method = d["method"].upper() if d["method"] else ""
    path   = (d["path"] or "/")
    return {
        "host": d["host"],
        "ts": ts,
        "method": method,
        "path": path,
        "status": status,
        "bytes": byt,
    }

def path_depth(p: str) -> int:
    p = p.split("?")[0]
    segs = [s for s in p.split("/") if s]
    return len(segs)

def is_static(p: str) -> bool:
    p = p.lower().split("?")[0]
    return p.endswith((".gif",".jpg",".jpeg",".png",".css",".js",".ico",".svg",".bmp",".woff",".woff2",".ttf",".map"))

def entropy(counter: Counter) -> float:
    N = sum(counter.values())
    if N == 0: return 0.0
    return -sum((c/N) * math.log2(c/N) for c in counter.values())

FEATURES = [
    'req_count','unique_hosts','status_2xx','status_3xx','status_4xx','status_5xx',
    'error_rate','bytes_sum','bytes_mean',
    'method_GET','method_POST','method_HEAD',
    'path_depth_mean','resource_entropy','host_entropy','static_ratio','robots_ratio'
]

def to_minute_features(parsed_rows):
    buckets = defaultdict(list)
    for r in parsed_rows:
        minute = r["ts"].replace(second=0, microsecond=0)
        buckets[minute].append(r)

    rows = []
    for minute, evs in buckets.items():
        req_count = len(evs)
        hosts   = [e["host"] for e in evs]
        methods = [e["method"] for e in evs]
        paths   = [e["path"] for e in evs]
        stats   = [e["status"] for e in evs]
        bytes_  = [e["bytes"]  for e in evs]

        s2 = sum(1 for s in stats if 200 <= s < 300)
        s3 = sum(1 for s in stats if 300 <= s < 400)
        s4 = sum(1 for s in stats if 400 <= s < 500)
        s5 = sum(1 for s in stats if 500 <= s < 600)

        rows.append({
            "minute": minute,
            "req_count": req_count,
            "unique_hosts": len(set(hosts)),
            "status_2xx": s2, "status_3xx": s3, "status_4xx": s4, "status_5xx": s5,
            "error_rate": ((s4+s5)/req_count) if req_count else 0.0,
            "bytes_sum": sum(bytes_),
            "bytes_mean": (sum(bytes_)/req_count) if req_count else 0.0,
            "method_GET": methods.count("GET")/req_count if req_count else 0.0,
            "method_POST": methods.count("POST")/req_count if req_count else 0.0,
            "method_HEAD": methods.count("HEAD")/req_count if req_count else 0.0,
            "path_depth_mean": float(np.mean([path_depth(p) for p in paths])) if req_count else 0.0,
            "resource_entropy": entropy(Counter([p.split("?")[0].rsplit("/",1)[-1] for p in paths])),
            "host_entropy": entropy(Counter(hosts)),
            "static_ratio": sum(1 for p in paths if is_static(p))/req_count if req_count else 0.0,
            "robots_ratio": sum(1 for p in paths if p.endswith("/robots.txt"))/req_count if req_count else 0.0,
        })

    df = pd.DataFrame(rows).sort_values("minute").reset_index(drop=True)
    return df
