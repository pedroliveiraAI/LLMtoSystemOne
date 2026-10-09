"""Evaluation: decision quality and behaviour preservation.

Decision metrics (paper Sec. 5, no post-hoc temperature scaling): accuracy, ECE (top-label
confidence, 10 equal-width bins, as in the JevBench harness), multi-class Brier.

Behaviour checks:
* ``post_bracket``: greedy-continue the Jev prompt and test whether the model stops right
  after ``k]`` (the paper reports 69/231 run-ons without KL anchors).
* ``conversation``: Dolly-15k (50 per category) and MT-Bench (80 two-turn) replies; rate of
  replies that fail to end within the token limit and of repetition loops (>= half of word
  4-grams repeated), optionally compared with a base model's replies.
"""

from __future__ import annotations

import json
import os
import random
import re
import time

import torch

from .data.hub import EVAL_SETS
from .data.schema import Example
from .jev import Jev
from .prompt import PREFILL, render

N_BINS = 10


def ece(conf: list[float], correct: list[bool], n_bins: int = N_BINS) -> float:
    """Top-label ECE with the JevBench harness binning (equal width, last bin closed)."""
    sums = [[0, 0.0, 0] for _ in range(n_bins)]  # n, confidence sum, correct
    for c, ok in zip(conf, correct):
        c = min(max(float(c), 0.0), 1.0)
        b = sums[min(int(c * n_bins), n_bins - 1)]
        b[0] += 1
        b[1] += c
        b[2] += int(ok)
    n = max(1, len(conf))
    return sum(k / n * abs(r / k - cs / k) for k, cs, r in sums if k)


def brier(probs: list[list[float]], gold: list[int]) -> float:
    return sum(sum((p - (1.0 if k == g else 0.0)) ** 2 for k, p in enumerate(ps))
               for ps, g in zip(probs, gold)) / len(gold)


def metrics(probs: list[list[float]], gold: list[int]) -> dict:
    pred = [max(range(len(p)), key=p.__getitem__) for p in probs]
    correct = [a == b for a, b in zip(pred, gold)]
    conf = [max(p) for p in probs]
    return {"n": len(gold), "accuracy": sum(correct) / len(gold), "ece": ece(conf, correct),
            "brier": brier(probs, gold)}


def evaluate_set(jev: Jev, examples: list[Example], readouts=("jev", "letters"), out_path: str | None = None,
                 log=print) -> dict:
    res = {r: {"probs": [], "gold": []} for r in readouts}
    fh = open(out_path, "w", encoding="utf-8") if out_path else None
    t0 = time.perf_counter()
    for i, e in enumerate(examples):
        row = {"i": i, "source": e.source, "gold": e.gold, "n_options": len(e.options)}
        for r in readouts:
            if r == "letters" and len(e.options) > 26:
                continue
            p = jev.decide(e.state, e.options, e.instruction) if r == "jev" else \
                jev.decide_letters(e.state, e.options, e.instruction)
            res[r]["probs"].append(p)
            res[r]["gold"].append(e.gold)
            row[r] = p
        if fh:
            fh.write(json.dumps(row) + "\n")
        if (i + 1) % 25 == 0:
            acc = sum(max(range(len(p)), key=p.__getitem__) == g
                      for p, g in zip(res[readouts[0]]["probs"], res[readouts[0]]["gold"])) / (i + 1)
            log(f"  {i + 1}/{len(examples)}  acc[{readouts[0]}]={acc:.3f}  {time.perf_counter() - t0:.0f}s")
    if fh:
        fh.close()
    return {r: metrics(v["probs"], v["gold"]) for r, v in res.items() if v["gold"]}


# --------------------------------------------------------------------------------------------
# Behaviour checks
# --------------------------------------------------------------------------------------------

@torch.no_grad()
def post_bracket(jev: Jev, examples: list[Example], max_new_tokens: int = 12) -> dict:
    """What does greedy decoding do right after the bracketed identifier?

    ``stop``: ends the reply; ``echo``: restates the chosen option's text (natural for chat
    models, harmless for a Jev); ``other``: anything else (drift). ``run_on`` = echo + other.
    Compare against the base model: the KL anchors aim to keep this profile unchanged."""
    counts = {"stop": 0, "echo": 0, "other": 0, "no_bracket": 0}
    for e in examples:
        text = render(jev.tok, e.state, e.options, e.instruction, jev.system).text
        ids = jev.tok(text, return_tensors="pt", add_special_tokens=False).to(jev.model.device)
        out = jev.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=jev.tok.pad_token_id)
        cont = jev.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.match(r"\s*(\d+)\]", cont)
        if not m:
            counts["no_bracket"] += 1
            continue
        after = cont[m.end():].strip()
        k = int(m.group(1)) - 1
        opt = e.options[k].strip() if 0 <= k < len(e.options) else ""
        if not after:
            counts["stop"] += 1
        elif opt and (opt.lower().startswith(after.lower()[:20]) or after.lower().startswith(opt.lower()[:20])):
            counts["echo"] += 1
        else:
            counts["other"] += 1
    n = max(1, len(examples))
    return {"n": len(examples), **counts, "run_on": counts["echo"] + counts["other"],
            "run_on_rate": (counts["echo"] + counts["other"]) / n, "drift_rate": counts["other"] / n,
            "prefill": PREFILL}


def _repetition_loop(text: str) -> bool:
    w = re.findall(r"\w+", text.lower())
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(grams) >= 8 and (len(grams) - len(set(grams))) / len(grams) >= 0.5


@torch.no_grad()
def _generate(jev: Jev, messages: list[dict], max_new_tokens: int) -> tuple[str, bool]:
    kwargs = {"enable_thinking": False} if "enable_thinking" in str(jev.tok.chat_template or "") else {}
    if jev.tok.chat_template:
        text = jev.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    else:
        text = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in messages) + "Assistant: "
    ids = jev.tok(text, return_tensors="pt", add_special_tokens=False).to(jev.model.device)
    out = jev.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0,
                             pad_token_id=jev.tok.pad_token_id)
    new = out[0, ids["input_ids"].shape[1]:]
    eos = set(jev.model.generation_config.eos_token_id if isinstance(jev.model.generation_config.eos_token_id, list)
              else [jev.model.generation_config.eos_token_id])
    ended = len(new) < max_new_tokens or int(new[-1]) in eos
    return jev.tok.decode(new, skip_special_tokens=True), ended


def conversation_prompts(n_per_category: int = 50, seed: int = 0, mt_bench: bool = True) -> list[dict]:
    from datasets import load_dataset
    rng = random.Random(seed)
    dolly = [r for r in load_dataset("databricks/databricks-dolly-15k", split="train")
             if len(r["instruction"]) + len(r["context"]) <= 4000]
    rng.shuffle(dolly)
    seen, per_cat, items = set(), {}, []
    for r in dolly:
        key = r["instruction"].strip().lower()
        if key in seen or per_cat.get(r["category"], 0) >= n_per_category:
            continue
        seen.add(key)
        per_cat[r["category"]] = per_cat.get(r["category"], 0) + 1
        msg = r["instruction"] + (f"\n\n{r['context']}" if r["context"] else "")
        items.append({"id": f"dolly/{r['category']}/{len(items)}", "turns": [msg]})
    if mt_bench:
        for r in load_dataset("HuggingFaceH4/mt_bench_prompts", split="train"):
            items.append({"id": f"mtbench/{r['prompt_id']}", "turns": list(r["prompt"])})
    return items


def conversation(jev: Jev, items: list[dict], max_new_tokens: int = 1024, out_path: str | None = None,
                 base_path: str | None = None, log=print) -> dict:
    replies = []
    for n, it in enumerate(items, 1):
        msgs = []
        for turn in it["turns"]:  # second MT-Bench turn follows the model's own first reply
            msgs.append({"role": "user", "content": turn})
            text, ended = _generate(jev, msgs, max_new_tokens)
            msgs.append({"role": "assistant", "content": text})
            replies.append({"id": it["id"], "turn": len(msgs) // 2, "reply": text, "ended": ended,
                            "loop": _repetition_loop(text)})
        if n % 10 == 0:
            log(f"  conversation {n}/{len(items)}")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            for r in replies:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    out = {"n_replies": len(replies),
           "unterminated_rate": sum(not r["ended"] for r in replies) / max(1, len(replies)),
           "loop_rate": sum(r["loop"] for r in replies) / max(1, len(replies))}
    if base_path and os.path.exists(base_path):
        with open(base_path, encoding="utf-8") as fh:
            base = {(d["id"], d["turn"]): d["reply"] for d in map(json.loads, fh)}
        same, jac = [], []
        for r in replies:
            b = base.get((r["id"], r["turn"]))
            if b is None:
                continue
            same.append(r["reply"].strip() == b.strip())
            A, B = set(re.findall(r"\w+", r["reply"].lower())), set(re.findall(r"\w+", b.lower()))
            jac.append(len(A & B) / max(1, len(A | B)))
        if same:
            out["identical_to_base"] = sum(same) / len(same)
            out["word_jaccard_to_base"] = sum(jac) / len(jac)
    return out


def run_eval(jev: Jev, tasks: list[str], limit: int | None, out_dir: str, readouts=("jev", "letters"),
             seed: int = 0, log=print) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for t in tasks:
        exs = EVAL_SETS[t]()
        exs = [e for e in exs if e.gold >= 0]
        if limit and len(exs) > limit:
            exs = random.Random(seed).sample(exs, limit)
        log(f"[eval] {t}: {len(exs)} examples")
        summary[t] = evaluate_set(jev, exs, readouts, os.path.join(out_dir, f"{t}.jsonl"), log)
        log(f"[eval] {t}: " + json.dumps(summary[t]))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def fmt_table(summary: dict) -> str:
    lines = ["| task | readout | n | acc | ECE | Brier |", "|---|---|---|---|---|---|"]
    for t, rs in summary.items():
        for r, m in rs.items():
            if isinstance(m, dict) and "accuracy" in m:
                lines.append(f"| {t} | {r} | {m['n']} | {100 * m['accuracy']:.1f} | {m['ece']:.3f} | {m['brier']:.3f} |")
    return "\n".join(lines)


__all__ = ["metrics", "ece", "brier", "evaluate_set", "post_bracket", "conversation", "conversation_prompts",
           "run_eval", "fmt_table"]
