"""One decision example, and the user's own JSONL format."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass


@dataclass
class Example:
    state: str
    options: list[str]
    gold: int                      # 0-based index into options
    instruction: str | None = None
    source: str = ""

    def shuffled(self, rng: random.Random) -> "Example":
        order = list(range(len(self.options)))
        rng.shuffle(order)
        return Example(self.state, [self.options[i] for i in order], order.index(self.gold),
                       self.instruction, self.source)


def read_jsonl(path: str, source: str | None = None) -> list[Example]:
    """``{"state": str, "options": [str, ...], "answer": int, "instruction": str?}`` per line."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            gold = d.get("answer", d.get("gold"))
            if not isinstance(gold, int) or not 0 <= gold < len(d["options"]):
                raise ValueError(f"{path}:{n}: 'answer' must be a 0-based index into 'options'")
            out.append(Example(d["state"], list(d["options"]), gold, d.get("instruction"), source or path))
    return out


def write_jsonl(examples: list[Example], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for e in examples:
            d = asdict(e)
            d["answer"] = d.pop("gold")
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
