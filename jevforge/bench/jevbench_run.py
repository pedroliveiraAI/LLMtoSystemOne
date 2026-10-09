"""Run the official JevBench harness (github.com/fstandhartinger/jevbench, pinned commit
1bcc55eb6c8cffde2306b3db03ede39b61c6152a) with a jevforge model as an in-process adapter.

The harness is used unmodified: its own task loader, Runner (raw evidence, ledger) and
summarize (accuracy, schema validity, Brier, ECE, ordinal MAE, paraphrase consistency).
Two shims only, both outside the harness source:
* ``fcntl`` does not exist on Windows; a no-op stand-in is injected (runs are single-process,
  so the ledger lock has nothing to protect).
* The harness CLI has a fixed adapter list, so we drive Runner/summarize directly instead.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types

JEVBENCH_COMMIT = "1bcc55eb6c8cffde2306b3db03ede39b61c6152a"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "jevbench"))
PUBLIC = [os.path.join(REPO, "datasets", "public", f"{n}.jsonl") for n in ("original", "easy", "hard")]


def _import_harness():
    if "fcntl" not in sys.modules:
        try:
            import fcntl  # noqa: F401
        except ImportError:
            shim = types.ModuleType("fcntl")
            shim.LOCK_SH, shim.LOCK_EX, shim.LOCK_UN = 1, 2, 8
            shim.flock = lambda *a, **k: None
            sys.modules["fcntl"] = shim
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import jevbench  # noqa: F401
    from jevbench.adapters.base import DecisionResult
    from jevbench.budget import Ledger
    from jevbench.runner import Runner
    from jevbench.summarize import public_export, summarize
    from jevbench.tasks import dataset_hash, load_jsonl
    return DecisionResult, Ledger, Runner, summarize, public_export, dataset_hash, load_jsonl


class JevforgeAdapter:
    """Native probabilities over the item's exact label set, read from next-token
    probabilities of the bracketed identifiers (nothing is generated)."""

    name = "jevforge"
    cost_basis = "local_no_provider_tariff"
    price_input_per_m = None
    price_output_per_m = None

    def __init__(self, jev, model_label: str, DecisionResult):
        self.jev = jev
        self.model = model_label
        self._DR = DecisionResult

    def run(self, task):
        from ..data.hub import jevbench_task_to_example
        res = self._DR(adapter=self.name, ok=False, probs_source="native", model=self.model)
        rec = {"state": task.state, "question": task.question, "labels": task.labels,
               "expected": task.expected, "family": task.family}
        ex, labels = jevbench_task_to_example(rec)
        res.request_body = {"question": task.question, "labels": labels}
        t0 = time.perf_counter()
        try:
            enc = self.jev.encode(ex.state, ex.options, ex.instruction)
            p = self.jev.decide(ex.state, ex.options, ex.instruction)
        except Exception as e:  # noqa: BLE001 - recorded as a failed attempt
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            return res
        res.latency_s = time.perf_counter() - t0
        res.probs = dict(zip(labels, p))
        res.usage = {"input_tokens": len(enc.prompt_ids), "output_tokens": 0}
        res.raw = {"answer": {"labels": labels, "probabilities": p},
                   "runtime": {"readout": "bracketed-id next-token probabilities (LLM-as-Jev)",
                               "scoring_mode": self.jev.mode, "prompt_tokens": len(enc.prompt_ids),
                               "probability_origin": "native-softmax"}}
        res.ok = True
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0


def headline_ece(summary: dict) -> float | None:
    e = summary.get("ece")
    return e.get("ece") if isinstance(e, dict) else e


def run_jevbench(jev, model_label: str, out_dir: str, tasks_paths: list[str] | None = None,
                 limit: int | None = None, log=print) -> dict:
    DecisionResult, Ledger, Runner, summarize, public_export, dataset_hash, load_jsonl = _import_harness()
    tasks = []
    for p in tasks_paths or PUBLIC:
        tasks += load_jsonl(p)
    if limit:
        tasks = tasks[:limit]
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    results = os.path.join(out_dir, "results.jsonl")
    for f in (results, os.path.join(out_dir, "ledger.jsonl")):
        if os.path.exists(f):
            os.remove(f)  # harness opens results with mode 'x'; a rerun replaces the old run
    raw_dir = os.path.join(out_dir, "raw")
    if os.path.isdir(raw_dir):
        import shutil
        shutil.rmtree(raw_dir)
    adapter = JevforgeAdapter(jev, model_label, DecisionResult)
    ledger = Ledger(os.path.join(out_dir, "ledger.jsonl"), cap_usd=1.0)
    runner = Runner(adapter, ledger, raw_dir=raw_dir, default_reserve_usd=0.0)
    log(f"[jevbench] {len(tasks)} tasks, harness commit {JEVBENCH_COMMIT[:7]}, model={model_label}")
    t0 = time.perf_counter()
    records = runner.run_all(tasks, results_path=results, progress_every=25)
    summary = summarize(tasks, records, None)
    summary["harness_commit"] = JEVBENCH_COMMIT
    summary["dataset_hash"] = dataset_hash(tasks)
    summary["wall_s"] = time.perf_counter() - t0
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    export = public_export(summary, tasks, records)
    export["dataset_hash"] = dataset_hash(tasks)
    with open(os.path.join(out_dir, "public_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(export, fh, indent=2, sort_keys=True, ensure_ascii=False)
    log(f"[jevbench] accuracy={summary['accuracy']:.4f} ({summary['n_correct']}/{summary['n_scorable']}) "
        f"ece={headline_ece(summary):.4f} brier={summary['brier_mean']:.4f} "
        f"macro_family={summary['macro_accuracy']:.4f} schema_validity={summary['schema_validity']:.3f}")
    return summary
