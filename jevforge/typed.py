"""TypeSafe-style typed decisions on top of a Jev.

A typed question is ``{"type": "choice"|"noul"|"score", "instructions": str, "criteria": ...}``:

* choice: ``criteria`` = ``{label: description}`` (or a list of labels)
          -> ``{"type": "choice", "choice": label, "probabilities": {label: p}}``
* noul:   ``criteria`` = ``{"true": text, "false": text}`` (optional)
          -> ``{"type": "noul", "noul": p_yes}``
* score:  ``criteria`` = ``[level_0_text, level_1_text, ...]``
          -> ``{"type": "score", "score": i, "probabilities": {"0": p, ...}}``

This is the answer shape of the TypeSafe ``/v1/systemone`` API (and of the JevBench
``typesafe`` adapter), so a jevforge server can stand in for it unchanged.
"""

from __future__ import annotations

import json

QUESTION_TYPES = ("choice", "noul", "score")


def question_to_options(question: dict) -> tuple[list[str], list[str]]:
    """(option texts shown to the model, labels returned to the caller), aligned."""
    qtype, crit = question.get("type"), question.get("criteria")
    if qtype == "noul":
        crit = crit or {}
        return ([f"No: {crit.get('false', 'No')}", f"Yes: {crit.get('true', 'Yes')}"], ["no", "yes"])
    if qtype == "score":
        if not isinstance(crit, list) or not crit:
            raise ValueError("score criteria must be a non-empty list of level descriptions")
        return [f"Level {i}: {lvl}" for i, lvl in enumerate(crit)], [str(i) for i in range(len(crit))]
    if qtype == "choice":
        if isinstance(crit, list):
            crit = {str(k): None for k in crit}
        if not isinstance(crit, dict) or not crit:
            raise ValueError("choice criteria must be a non-empty {label: description} map or list")
        labels = [str(k) for k in crit]
        return [f"{k}: {v}" if v else k for k, v in crit.items()], labels
    raise ValueError(f"question type must be one of {QUESTION_TYPES}, got {qtype!r}")


def state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def answer(jev, state, question: dict) -> dict:
    """One typed answer, with native probabilities over the exact label set."""
    options, labels = question_to_options(question)
    probs = jev.decide(state_text(state), options, question.get("instructions"))
    qtype = question["type"]
    if qtype == "noul":
        return {"type": "noul", "noul": probs[1]}
    by_label = dict(zip(labels, probs))
    best = max(by_label, key=by_label.get)
    if qtype == "score":
        return {"type": "score", "score": int(best), "probabilities": by_label}
    return {"type": "choice", "choice": best, "probabilities": by_label}


def answer_all(jev, state, questions: dict) -> dict:
    """``{name: question}`` -> ``{name: answer}``, as in a /v1/systemone request."""
    return {name: answer(jev, state, q) for name, q in questions.items()}
