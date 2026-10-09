"""LoRA fine-tuning with the LLM-as-Jev objective (paper Sec. 4, Table 3).

Defaults: LoRA r=16, alpha=32, dropout 0 on every linear layer of the decoder; AdamW
(lr 1e-4, weight decay 0.01), grad clip 1.0, linear warmup over 10% of steps then linear
decay to 10% of peak, one epoch, global batch 32, shuffled option order, lambda=1.
The frozen reference p0 is the same network with the adapter disabled, so no second copy
of the weights is held in memory. Training stops if the root legal log-mass drops by more
than 0.5 nats relative to the reference (paper's health check).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass

import torch

from .data.schema import Example
from .jev import Jev, load_model_and_tokenizer
from .losses import jev_loss
from .prompt import render
from .scorer import FORWARDS
from .trie import encode


@dataclass
class TrainConfig:
    model: str
    out_dir: str
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: str = "all-linear"
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_frac: float = 0.10
    final_lr_frac: float = 0.10
    epochs: int = 1
    batch_size: int = 32
    lam: float = 1.0
    max_len: int = 8192
    max_steps: int | None = None
    health_nats: float = 0.5
    grad_checkpointing: bool = False
    save_every: int = 50
    seed: int = 20260922
    device: str | None = None
    dtype: str | None = None
    revision: str | None = None


def lr_factor(step: int, total: int, warmup: int, final_frac: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    span = max(1, total - warmup)
    return 1.0 - (1.0 - final_frac) * min(1.0, (step - warmup) / span)


def train(cfg: TrainConfig, examples: list[Example], log=print) -> str:
    from peft import LoraConfig, get_peft_model

    os.makedirs(cfg.out_dir, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    model, tok = load_model_and_tokenizer(cfg.model, device=cfg.device, dtype=cfg.dtype, revision=cfg.revision)
    # Pick tree vs naive scoring once, on the base model, with the auto self-check.
    probe = Jev(model, tok)
    probe.decide("probe", ["a", "b", "c"])
    forward = FORWARDS[probe.mode]
    log(f"[train] scoring mode: {probe.mode}")

    targets = cfg.lora_targets if cfg.lora_targets == "all-linear" else [t.strip() for t in cfg.lora_targets.split(",")]
    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                             target_modules=targets, task_type="CAUSAL_LM"))
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[train] trainable params: {n_train / 1e6:.1f}M")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(examples) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    warmup = max(1, int(cfg.warmup_frac * total))
    with open(os.path.join(cfg.out_dir, "train_config.json"), "w", encoding="utf-8") as fh:
        json.dump({**asdict(cfg), "n_examples": len(examples), "total_steps": total,
                   "trainable_params": n_train}, fh, indent=2)
    log_fh = open(os.path.join(cfg.out_dir, "train_log.jsonl"), "a", encoding="utf-8")

    step, t0, stopped = 0, time.perf_counter(), None
    for epoch in range(cfg.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for b in range(0, len(order), cfg.batch_size):
            if step >= total:
                break
            for g in opt.param_groups:
                g["lr"] = cfg.lr * lr_factor(step, total, warmup, cfg.final_lr_frac)
            agg = {"loss": 0.0, "tree": 0.0, "mass": 0.0, "out": 0.0, "pos": 0.0, "root_delta": 0.0, "acc": 0.0}
            n = 0
            batch = [examples[i].shuffled(rng) for i in order[b:b + cfg.batch_size]]
            for e in batch:
                enc = encode(tok, render(tok, e.state, e.options, e.instruction), len(e.options))
                if len(enc.prompt_ids) + len(enc.trie.nodes) > cfg.max_len:
                    continue
                with torch.no_grad(), model.disable_adapter():
                    ref = forward(model, enc)
                cur = forward(model, enc)
                parts = jev_loss(cur, ref, enc, e.gold, cfg.lam, rng)
                (parts.total / len(batch)).backward()
                n += 1
                agg["loss"] += parts.total.item()
                for k in ("tree", "mass", "out", "pos"):
                    agg[k] += getattr(parts, k)
                agg["root_delta"] += parts.root_logmass - parts.root_logmass_ref
                agg["acc"] += float(parts.tree < math.log(2))  # q(gold) > 0.5
            if n == 0:
                continue
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rec = {k: v / n for k, v in agg.items()}
            rec.update(step=step, total=total, lr=opt.param_groups[0]["lr"], n=n,
                       elapsed_s=round(time.perf_counter() - t0, 1))
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()
            log(f"[train] step {step}/{total} loss={rec['loss']:.4f} tree={rec['tree']:.4f} "
                f"mass={rec['mass']:.2e} out={rec['out']:.2e} pos={rec['pos']:.2e} "
                f"root_dlogm={rec['root_delta']:+.3f} q>0.5={rec['acc']:.2f} lr={rec['lr']:.2e} "
                f"[{rec['elapsed_s']:.0f}s]")
            if rec["root_delta"] < -cfg.health_nats:
                stopped = f"health check: root legal log-mass fell {rec['root_delta']:.3f} nats at step {step}"
                log(f"[train] STOP: {stopped}")
                break
            if cfg.save_every and step % cfg.save_every == 0:
                model.save_pretrained(os.path.join(cfg.out_dir, "checkpoint-last"))
        if stopped or step >= total:
            break

    model.save_pretrained(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    with open(os.path.join(cfg.out_dir, "train_result.json"), "w", encoding="utf-8") as fh:
        json.dump({"steps": step, "stopped": stopped, "elapsed_s": time.perf_counter() - t0}, fh, indent=2)
    log_fh.close()
    log(f"[train] adapter saved to {cfg.out_dir}")
    return cfg.out_dir
