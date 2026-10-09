"""Per-decision inference latency, from the per-item JevBench results in runs/.

    python scripts/latency_report.py [runs/<run>/jevbench/results.jsonl ...]   -> markdown on stdout

The first item of each run is excluded from the statistics and reported apart: it
includes the one-time scoring-mode self-check (skip it in production with --mode).
"""

import glob
import json
import os
import statistics as st
import sys

BUCKETS = [(0, 300, "< 300"), (300, 800, "300–800"), (800, float("inf"), "> 800")]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def load(path):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    lat = [r["latency_s"] for r in rows]
    toks = [(r.get("usage") or {}).get("input_tokens", 0) for r in rows]
    meta_path = os.path.join(os.path.dirname(os.path.dirname(path)), "run.json")
    label = json.load(open(meta_path, encoding="utf-8")).get("label") if os.path.exists(meta_path) else path
    return label, lat[0], lat[1:], toks[1:]


def main(paths):
    runs = [load(p) for p in paths]
    out = ["| Model | median | mean | p95 | first request |", "|---|---|---|---|---|"]
    for label, first, lat, _ in runs:
        out.append(f"| {label} | **{st.median(lat):.2f} s** | {st.mean(lat):.2f} s | {pct(lat, .95):.2f} s | {first:.1f} s |")
    out += ["", "| Prompt length | " + " | ".join(r[0] for r in runs) + " |", "|---" * (len(runs) + 1) + "|"]
    for lo, hi, name in BUCKETS:
        cells = []
        for _, _, lat, toks in runs:
            b = [x for x, t in zip(lat, toks) if lo <= t < hi]
            cells.append(f"{st.median(b):.2f} s (n={len(b)})" if b else "—")
        out.append(f"| {name} tokens | " + " | ".join(cells) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob("runs/*-base/jevbench/results.jsonl")))
