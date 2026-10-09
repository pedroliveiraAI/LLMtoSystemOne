"""Merge a LoRA adapter into its base model, save it, optionally push it to the Hub."""

from __future__ import annotations

import json
import os

from .jev import load_model_and_tokenizer

CARD = """---
base_model: {base}
library_name: transformers
tags: [jev, decision-model, llm-as-jev, jevforge]
---

# {name}

A Jev-style decision model: given a state and K options it returns a probability
distribution over the options, read from next-token probabilities of bracketed numeric
identifiers (LLM-as-Jev, arXiv 2610.02076v2). Built with `jevforge` from `{base}`.

## Use

```python
from jevforge import Jev
jev = Jev.from_pretrained("{repo}")
jev.decide("A customer bought 12 days ago but has no receipt. Refund?",
           ["Refund", "Decline: missing receipt", "Escalate"], "Apply the 30-day receipt policy.")
```

Prompt contract: options listed as `[1] .. [K]`, user turn ends with
"Answer only with its bracketed numeric identifier.", assistant prefilled with
`Best answer: [`; option k is scored by the completion `k]`.

## Training

{training}

## Evaluation

{evaluation}
"""


def export(model_id: str, adapter: str, out_dir: str, push: str | None = None, private: bool = True,
           eval_table: str = "", log=print) -> str:
    model, tok = load_model_and_tokenizer(model_id, adapter=adapter, device="cpu")
    model = model.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    cfg_path = os.path.join(adapter, "train_config.json")
    training = "LoRA with the LLM-as-Jev tree-factorized listwise loss and KL anchors."
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as fh:
            training += "\n\n```json\n" + json.dumps(json.load(fh), indent=2) + "\n```"
    name = push or os.path.basename(os.path.normpath(out_dir))
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(CARD.format(base=model_id, name=name, repo=push or out_dir, training=training,
                             evaluation=eval_table or "See the jevforge run directory."))
    log(f"[export] merged model saved to {out_dir}")
    if push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(push, private=private, exist_ok=True)
        api.upload_folder(folder_path=out_dir, repo_id=push)
        log(f"[export] pushed to https://huggingface.co/{push} (private={private})")
    return out_dir
