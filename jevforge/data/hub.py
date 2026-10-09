"""Hugging Face dataset adapters into ``Example``.

Evaluation sets are the paper's external benchmarks (Sec. 5): BoolQ, MMLU, MMLU-Pro,
ARC-Challenge, WinoGrande, SciQ, Banking77. Training sources follow Table 4 of the paper
(intent set + reasoning sets).
"""

from __future__ import annotations

import json
import random
from typing import Callable

from .schema import Example

INTENT_INSTRUCTION = "Which intent does the user's message express?"


def _load(name, config=None, split="train"):
    from datasets import load_dataset
    return load_dataset(name, config, split=split, trust_remote_code=True)


def _labels_from_letters(keys, answer_key):
    return list(keys).index(answer_key)


def _intent_examples(rows, label_names: list[str], source: str, rng: random.Random,
                     k_options: int | None, text_key="text") -> list[Example]:
    """Intent routing. ``k_options=None`` presents every intent (evaluation); otherwise a
    random subset of size uniform in [2, k_options] containing the gold intent (training)."""
    names = [n.replace("_", " ") for n in label_names]
    out = []
    for r, gold_name in rows:
        if k_options is None:
            opts, gold = names, names.index(gold_name)
        else:
            K = rng.randint(2, min(k_options, len(names)))
            distract = rng.sample([n for n in names if n != gold_name], K - 1)
            opts = distract + [gold_name]
            rng.shuffle(opts)
            gold = opts.index(gold_name)
        out.append(Example(str(r[text_key]), list(opts), gold, INTENT_INSTRUCTION, source))
    return out


# --------------------------------------------------------------------------------------------
# Evaluation sets
# --------------------------------------------------------------------------------------------

def boolq(split="validation", **_):
    return [Example(f"{r['passage']}\n\nQuestion: {r['question']}?", ["No", "Yes"], int(bool(r["answer"])),
                    "Answer the question using the passage.", "boolq") for r in _load("google/boolq", None, split)]


def mmlu(split="test", **_):
    return [Example(r["question"], list(r["choices"]), int(r["answer"]),
                    f"Answer this {r['subject'].replace('_', ' ')} question.", "mmlu")
            for r in _load("cais/mmlu", "all", split)]


def mmlu_pro(split="test", n=800, seed=0, **_):
    rows = list(_load("TIGER-Lab/MMLU-Pro", None, split))
    random.Random(seed).shuffle(rows)  # paper: 800-question sample
    return [Example(r["question"], list(r["options"]), int(r["answer_index"]),
                    f"Answer this {r['category']} question.", "mmlu_pro") for r in rows[:n]]


def arc_challenge(split="test", **_):
    return [Example(r["question"], list(r["choices"]["text"]),
                    _labels_from_letters(r["choices"]["label"], r["answerKey"]),
                    "Answer the science question.", "arc_challenge")
            for r in _load("allenai/ai2_arc", "ARC-Challenge", split)]


def winogrande(split="validation", **_):
    return [Example(r["sentence"], [r["option1"], r["option2"]], int(r["answer"]) - 1,
                    "Which option correctly fills the blank (_)?", "winogrande")
            for r in _load("allenai/winogrande", "winogrande_xl", split)]


def sciq(split="test", seed=0, **_):
    rng, out = random.Random(seed), []
    for r in _load("allenai/sciq", None, split):
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        order = list(range(4))
        rng.shuffle(order)
        out.append(Example(r["question"], [opts[i] for i in order], order.index(0),
                           "Answer the science question.", "sciq"))
    return out


def banking77(split="test", k_options=None, seed=0, **_):
    ds = _load("PolyAI/banking77", None, split)
    names = ds.features["label"].names
    rows = [(r, names[int(r["label"])].replace("_", " ")) for r in ds]
    return _intent_examples(rows, names, "banking77", random.Random(seed), k_options)


# --------------------------------------------------------------------------------------------
# Training sources (paper Table 4)
# --------------------------------------------------------------------------------------------

def clinc150(split="train", k_options=100, seed=0, **_):
    ds = _load("clinc/clinc_oos", "plus", split)
    names = ds.features["intent"].names
    rows = [(r, names[int(r["intent"])].replace("_", " ")) for r in ds]
    return _intent_examples(rows, names, "clinc150", random.Random(seed), k_options)


def massive(split="train", k_options=60, seed=0, **_):
    ds = _load("mteb/amazon_massive_intent", "en", split)
    names = sorted(set(ds["label"]))
    rows = [(r, r["label"].replace("_", " ")) for r in ds]
    return _intent_examples(rows, names, "massive", random.Random(seed), k_options)


def bitext(split="train", k_options=27, seed=0, **_):
    ds = _load("bitext/Bitext-customer-support-llm-chatbot-training-dataset", None, split)
    names = sorted(set(ds["intent"]))
    rows = [(r, r["intent"].replace("_", " ")) for r in ds]
    return _intent_examples(rows, names, "bitext", random.Random(seed), k_options, text_key="instruction")


def commonsense_qa(split="train", **_):
    return [Example(r["question"], list(r["choices"]["text"]),
                    _labels_from_letters(r["choices"]["label"], r["answerKey"]),
                    "Choose the most sensible answer.", "commonsense_qa")
            for r in _load("tau/commonsense_qa", None, split) if r["answerKey"]]


def hellaswag(split="train", **_):
    return [Example(f"{r['activity_label']}: {r['ctx']}", list(r["endings"]), int(r["label"]),
                    "Which ending best continues the text?", "hellaswag")
            for r in _load("Rowan/hellaswag", None, split) if str(r["label"]).strip()]


def cosmos_qa(split="train", **_):
    return [Example(f"{r['context']}\n\nQuestion: {r['question']}",
                    [r[f"answer{i}"] for i in range(4)], int(r["label"]),
                    "Answer the question about the passage.", "cosmos_qa")
            for r in _load("allenai/cosmos_qa", None, split)]


def piqa(split="train", **_):
    return [Example(r["goal"], [r["sol1"], r["sol2"]], int(r["label"]),
                    "Which solution achieves the goal?", "piqa") for r in _load("baber/piqa", None, split)]


def anli_art(split="train", **_):
    return [Example(f"Beginning: {r['observation_1']}\nEnding: {r['observation_2']}",
                    [r["hypothesis_1"], r["hypothesis_2"]], int(r["label"]) - 1,
                    "Which middle best explains how the beginning leads to the ending?", "anli")
            for r in _load("allenai/art", None, split)]


def logiqa2(split="train", **_):
    out = []
    for r in _load("baber/logiqa2", None, split):
        opts = r["options"] if isinstance(r["options"], list) else json.loads(r["options"])
        out.append(Example(f"{r['text']}\n\nQuestion: {r['question']}", list(opts), int(r["answer"]),
                           "Choose the logically correct answer.", "logiqa2"))
    return out


def reclor(split="train", **_):
    return [Example(f"{r['context']}\n\nQuestion: {r['question']}", list(r["answers"]), int(r["label"]),
                    "Choose the logically correct answer.", "reclor") for r in _load("metaeval/reclor", None, split)]


def proofwriter(split="train", **_):
    labels = ["True", "False", "Unknown"]
    return [Example(f"Facts and rules: {r['theory']}\n\nStatement: {r['question']}", labels,
                    labels.index(r["answer"]), "Given only the facts and rules, is the statement true, false, or unknown?",
                    "proofwriter")
            for r in _load("tasksource/proofwriter", None, split) if r["answer"] in labels]


# --------------------------------------------------------------------------------------------
# JevBench public items as Examples (used for decontamination and quick local checks;
# official numbers come from the jevbench harness, see jevforge/bench).
# --------------------------------------------------------------------------------------------

def jevbench_task_to_example(t: dict) -> tuple[Example, list[str]]:
    """Map a JevBench record to (Example, labels aligned with options)."""
    from ..typed import question_to_options, state_text
    options, labels = question_to_options(t["question"])
    exp = str(t["expected"]) if t.get("expected") is not None else None
    gold = labels.index(exp) if exp in labels else -1
    return Example(state_text(t["state"]), options, gold, t["question"]["instructions"],
                   f"jevbench/{t.get('family', '')}"), labels


def jevbench_public(path: str | None = None, **_):
    import glob
    import os
    root = path or os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "jevbench", "datasets", "public")
    out = []
    for f in sorted(glob.glob(os.path.join(root, "*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            out += [jevbench_task_to_example(json.loads(line))[0] for line in fh if line.strip()]
    return out


EVAL_SETS: dict[str, Callable] = {
    "boolq": boolq, "mmlu": mmlu, "mmlu_pro": mmlu_pro, "arc_challenge": arc_challenge,
    "winogrande": winogrande, "sciq": sciq, "banking77": banking77, "jevbench_public": jevbench_public,
}
TRAIN_SOURCES: dict[str, Callable] = {
    "clinc150": clinc150, "massive": massive, "bitext": bitext, "commonsense_qa": commonsense_qa,
    "hellaswag": hellaswag, "cosmos_qa": cosmos_qa, "piqa": piqa, "anli": anli_art, "logiqa2": logiqa2,
    "reclor": reclor, "proofwriter": proofwriter,
}
MIXES = {
    "intent": ["clinc150", "massive", "bitext", "commonsense_qa", "hellaswag"],
    "reason": ["reclor", "logiqa2", "cosmos_qa", "piqa", "anli", "proofwriter"],
}
MIXES["general"] = MIXES["intent"] + MIXES["reason"]
# Paper: "Reason-9k" = 1.5k per reasoning set.
REASON_PER_SOURCE = 1500
