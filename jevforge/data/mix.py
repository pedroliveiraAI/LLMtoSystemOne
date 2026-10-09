"""Training mixture and decontamination.

Decontamination follows the paper: drop a training example that shares a 13-word
n-gram, or an identical sentence of six or more words, with any evaluation item.
"""

from __future__ import annotations

import random
import re
import sys

from .hub import EVAL_SETS, MIXES, TRAIN_SOURCES
from .schema import Example

_WORD = re.compile(r"\w+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _text(e: Example) -> str:
    return " ".join([e.state] + list(e.options))


def _ngrams(text: str, n: int = 13) -> set:
    w = _WORD.findall(text.lower())
    return {hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1)}


def _sentences(text: str, min_words: int = 6) -> set:
    out = set()
    for s in _SENT.split(text):
        w = _WORD.findall(s.lower())
        if len(w) >= min_words:
            out.add(hash(tuple(w)))
    return out


class Decontaminator:
    def __init__(self, eval_examples: list[Example]):
        self.grams, self.sents = set(), set()
        for e in eval_examples:
            t = _text(e)
            self.grams |= _ngrams(t)
            self.sents |= _sentences(t)

    def is_contaminated(self, e: Example) -> bool:
        t = _text(e)
        return bool(_ngrams(t) & self.grams) or bool(_sentences(t) & self.sents)


def load_eval_pool(names: list[str]) -> list[Example]:
    pool = []
    for n in names:
        try:
            pool += EVAL_SETS[n]()
        except Exception as e:  # noqa: BLE001 - one missing benchmark must not block training
            print(f"[mix] decontamination: could not load {n}: {e}", file=sys.stderr)
    return pool


def build_mix(sources: list[str] | str = "general", max_examples: int | None = None, seed: int = 20260922,
              decontaminate: list[str] | None = None, extra: list[Example] | None = None,
              log=print) -> list[Example]:
    """Equal quota per source when ``max_examples`` is set (stratified), else everything
    (with the paper's 1.5k cap per reasoning source)."""
    if isinstance(sources, str):
        sources = MIXES.get(sources, [s.strip() for s in sources.split(",") if s.strip()])
    rng = random.Random(seed)
    deco = Decontaminator(load_eval_pool(decontaminate)) if decontaminate else None
    per_source = max_examples // max(1, len(sources) + (1 if extra else 0)) if max_examples else None
    out: list[Example] = []
    groups = [(s, None) for s in sources] + ([("custom", extra)] if extra else [])
    for name, rows in groups:
        try:
            rows = rows if rows is not None else TRAIN_SOURCES[name](seed=seed)
        except Exception as e:  # noqa: BLE001
            log(f"[mix] skipping {name}: {type(e).__name__}: {e}")
            continue
        rng.shuffle(rows)
        cap = per_source
        if cap is None and name in MIXES["reason"]:
            from .hub import REASON_PER_SOURCE
            cap = REASON_PER_SOURCE
        kept, dropped = [], 0
        for e in rows:
            if cap is not None and len(kept) >= cap:
                break
            if deco and deco.is_contaminated(e):
                dropped += 1
                continue
            kept.append(e)
        log(f"[mix] {name}: {len(kept)} examples ({dropped} dropped as contaminated)")
        out += kept
    rng.shuffle(out)
    return out
