"""Comparison table: your runs next to published Jev-class numbers.

Reference rows are *cited*, not re-run here. Sources:
* paper: LLM-as-Jev, arXiv 2610.02076v2, main results table (JevBench public-231 accuracy,
  Banking77 accuracy, JevBench ECE).
* Open-Jev benchmark page (zefan-cai.github.io/open-jev/benchmarks), public-231 correct counts,
  historical JevBench commit f8ce713.
Different harness versions and prompts make cited rows indicative, not strictly comparable;
the paper reports the same public 231 items and the same accuracy definition.
"""

from __future__ import annotations

import json
import os

REFERENCES = [
    # name, jevbench_acc, banking77_acc, jevbench_ece, source
    ("Jev 1.13 (hosted, proprietary)", 86.6, None, None, "Open-Jev page: 200/231"),
    ("Winnow-12B", 86.6, None, 0.058, "paper"),
    ("Open-Jev-27B v1.1", 85.3, None, None, "Open-Jev page: 197/231"),
    ("Paper: Qwen3.5-4B LoRA", 84.0, 75.4, 0.050, "paper"),
    ("Paper: Qwen3.5-4B training-free", 81.4, 69.0, 0.057, "paper"),
    ("SemIf (Qwen3.5-4B, letter logits)", 80.5, None, 0.061, "paper"),
    ("Paper: Qwen3-0.6B LoRA", 61.0, 59.0, 0.121, "paper"),
    ("Paper: Qwen3-0.6B training-free", 56.7, 22.2, 0.278, "paper"),
]


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def collect_run(run_dir: str) -> dict:
    """A run dir may hold jevbench/summary.json and/or eval/summary.json."""
    row = {"name": os.path.basename(os.path.normpath(run_dir))}
    jb = os.path.join(run_dir, "jevbench", "summary.json")
    if os.path.exists(jb):
        s = _load(jb)
        e = s.get("ece")
        row.update(jevbench=100 * s["accuracy"], jevbench_n=f"{s['n_correct']}/{s['n_scorable']}",
                   jevbench_ece=e.get("ece") if isinstance(e, dict) else e, jevbench_brier=s.get("brier_mean"))
    ev = os.path.join(run_dir, "eval", "summary.json")
    if os.path.exists(ev):
        for task, rs in _load(ev).items():
            if "jev" in rs:
                row[task] = 100 * rs["jev"]["accuracy"]
    meta = os.path.join(run_dir, "run.json")
    if os.path.exists(meta):
        m = _load(meta)
        row["name"] = m.get("label", row["name"])
    return row


def table(run_dirs: list[str], tasks: list[str] | None = None) -> str:
    runs = [collect_run(d) for d in run_dirs]
    tasks = tasks or sorted({k for r in runs for k in r} - {"name", "jevbench", "jevbench_n", "jevbench_ece",
                                                            "jevbench_brier"})
    head = ["system", "JevBench-231 acc", "JevBench ECE"] + tasks + ["source"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    fmt = lambda v, p=1: "" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))  # noqa: E731
    for r in runs:
        jb = r.get("jevbench")
        jb_s = f"{jb:.1f} ({r['jevbench_n']})" if jb is not None else ""
        lines.append("| " + " | ".join([f"**{r['name']}**", jb_s, fmt(r.get("jevbench_ece"), 3)]
                                       + [fmt(r.get(t)) for t in tasks] + ["this run"]) + " |")
    for name, jb, b77, e, src in REFERENCES:
        cells = [fmt(b77) if t == "banking77" else "" for t in tasks]
        lines.append("| " + " | ".join([name, fmt(jb), fmt(e, 3)] + cells + [f"cited ({src})"]) + " |")
    return "\n".join(lines)
