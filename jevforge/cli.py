"""jevforge command line.

  jevforge inspect   --model M                       is this model usable as a Jev?
  jevforge eval      --model M [--adapter A] --tasks arc_challenge,banking77 --out runs/x
  jevforge jevbench  --model M [--adapter A] --out runs/x      official JevBench public-231
  jevforge behaviour --model M [--adapter A] --out runs/x      run-on after "]", conversation
  jevforge train     --model M --mix general --max-examples 4000 --out runs/x
  jevforge compare   runs/base runs/x                          table vs published Jev numbers
  jevforge export    --model M --adapter runs/x --out merged/ [--push user/name]
  jevforge serve     --model M [--adapter A]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

DEFAULT_EVAL = "boolq,mmlu,mmlu_pro,arc_challenge,winogrande,sciq,banking77"
DEFAULT_DECONTAM = DEFAULT_EVAL + ",jevbench_public"


def _jev(args):
    from .jev import Jev
    return Jev.from_pretrained(args.model, adapter=getattr(args, "adapter", None), device=args.device,
                               dtype=args.dtype, revision=args.revision, mode=args.mode)


def _write_run_meta(out: str, args, extra: dict | None = None):
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "run.json")
    meta = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
    meta.update({"model": args.model, "adapter": getattr(args, "adapter", None),
                 "label": getattr(args, "label", None) or meta.get("label") or os.path.basename(os.path.normpath(out))})
    meta.update(extra or {})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def cmd_inspect(args):
    import torch
    from .scorer import FORWARDS, legal_logmass
    jev = _jev(args)
    print(f"model: {args.model}  device: {jev.model.device}  dtype: {next(jev.model.parameters()).dtype}")
    print(f"chat template: {'yes' if jev.tok.chat_template else 'no (plain fallback)'}")
    for K in (4, 12, 100):
        opts = [f"option {i}" for i in range(1, K + 1)]
        enc = jev.encode("Example state.", opts, "Pick one.")
        p = jev.decide("Example state.", opts, "Pick one.")  # also settles tree vs naive mode
        with torch.no_grad():
            nlp = FORWARDS[jev.mode](jev.model, enc)
        m0 = legal_logmass(nlp, enc, 0).exp().item()
        toks = [jev.tok.convert_ids_to_tokens(s) for s in (enc.suffixes[0], enc.suffixes[-1])]
        print(f"K={K:<4} prompt_tokens={len(enc.prompt_ids):<5} trie_depth={enc.trie.depth} "
              f"nodes={len(enc.trie.nodes):<4} root_legal_mass={m0:.4f} suffix[1]={toks[0]} suffix[K]={toks[1]} "
              f"max_p={max(p):.3f}")
    print("prompt tail:", repr(jev.tok.decode(jev.encode("s", ["a", "b"]).prompt_ids[-30:])))
    print(f"scoring mode: {jev.mode}")
    print("A root legal mass near 1 means the model already answers with a bracketed id (paper Sec. 3).")


def cmd_eval(args):
    from .evaluate import fmt_table, run_eval
    jev = _jev(args)
    out = os.path.join(args.out, "eval")
    _write_run_meta(args.out, args)
    s = run_eval(jev, [t.strip() for t in args.tasks.split(",") if t.strip()], args.limit, out,
                 tuple(args.readouts.split(",")), args.seed)
    print(fmt_table(s))


def cmd_jevbench(args):
    from .bench.jevbench_run import run_jevbench
    jev = _jev(args)
    _write_run_meta(args.out, args)
    label = args.label or (f"{args.model}+{args.adapter}" if args.adapter else args.model)
    run_jevbench(jev, label, os.path.join(args.out, "jevbench"), limit=args.limit)


def cmd_behaviour(args):
    import random
    from .data.hub import jevbench_public
    from .evaluate import conversation, conversation_prompts, post_bracket
    jev = _jev(args)
    _write_run_meta(args.out, args)
    out = os.path.join(args.out, "behaviour")
    os.makedirs(out, exist_ok=True)
    res = {}
    if args.post_bracket:
        exs = jevbench_public()
        if args.post_bracket < len(exs):
            exs = random.Random(0).sample(exs, args.post_bracket)
        res["post_bracket"] = post_bracket(jev, exs)
        print("[behaviour] post-bracket:", res["post_bracket"])
    if args.conversation:
        items = conversation_prompts(args.n_per_category, mt_bench=not args.no_mt_bench)
        base = os.path.join(args.base, "behaviour", "conversation.jsonl") if args.base else None
        res["conversation"] = conversation(jev, items, args.max_new_tokens,
                                           os.path.join(out, "conversation.jsonl"), base)
        print("[behaviour] conversation:", res["conversation"])
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)


def cmd_train(args):
    from .data.mix import build_mix
    from .data.schema import read_jsonl, write_jsonl
    from .train import TrainConfig, train
    extra = read_jsonl(args.train_file) if args.train_file else None
    sources = args.mix if args.mix != "none" else []
    decontam = [d for d in args.decontaminate.split(",") if d] if args.decontaminate != "none" else None
    examples = build_mix(sources, args.max_examples, args.seed, decontam, extra) if sources else (extra or [])
    if not examples:
        sys.exit("no training examples")
    os.makedirs(args.out, exist_ok=True)
    write_jsonl(examples, os.path.join(args.out, "train_data.jsonl"))
    cfg = TrainConfig(model=args.model, out_dir=args.out, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_targets=args.lora_targets, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                      lam=args.lam, max_len=args.max_len, max_steps=args.max_steps, grad_checkpointing=args.grad_ckpt,
                      seed=args.seed, device=args.device, dtype=args.dtype, revision=args.revision,
                      save_every=args.save_every)
    _write_run_meta(args.out, args, {"adapter": args.out, "trained": True})
    train(cfg, examples)


def cmd_compare(args):
    from .compare import table
    print(table(args.runs, [t for t in (args.tasks or "").split(",") if t] or None))


def cmd_export(args):
    from .export import export
    export(args.model, args.adapter, args.out, args.push, private=not args.public)


def cmd_serve(args):
    from .serve import serve
    name = args.label or (f"{args.model}+{os.path.basename(os.path.normpath(args.adapter))}" if args.adapter else args.model)
    serve(_jev(args), args.host, args.port, model_name=name)


def cmd_general(args):
    """General capability (paper: GSM8K, IFEval, TriviaQA, LAMBADA; WikiText-2 ppl) via lm-eval."""
    model_args = f"pretrained={args.model}" + (f",peft={args.adapter}" if args.adapter else "")
    out = os.path.join(args.out, "general")
    cmd = [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args", model_args,
           "--tasks", args.tasks, "--output_path", out, "--batch_size", "1", "--apply_chat_template"]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    print(" ".join(cmd))
    sys.exit(subprocess.call(cmd))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jevforge", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p, adapter=True):
        p.add_argument("--model", required=True, help="Hugging Face model id or local path")
        if adapter:
            p.add_argument("--adapter", default=None, help="LoRA adapter dir (from `jevforge train`)")
        p.add_argument("--device", default=None)
        p.add_argument("--dtype", default=None, help="float32 | bfloat16 | float16")
        p.add_argument("--revision", default=None)
        p.add_argument("--mode", default="auto", choices=["auto", "tree", "cache", "naive"])
        p.add_argument("--label", default=None, help="name of this system in tables")

    p = sub.add_parser("inspect")
    model_args(p)
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("eval")
    model_args(p)
    p.add_argument("--tasks", default=DEFAULT_EVAL)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--readouts", default="jev,letters")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("jevbench")
    model_args(p)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_jevbench)

    p = sub.add_parser("behaviour")
    model_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--post-bracket", type=int, default=231, help="JevBench items to greedy-continue (0 = skip)")
    p.add_argument("--conversation", action="store_true", help="Dolly-15k + MT-Bench replies (slow on CPU)")
    p.add_argument("--n-per-category", type=int, default=50)
    p.add_argument("--no-mt-bench", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--base", default=None, help="run dir of the base model, to compare replies")
    p.set_defaults(fn=cmd_behaviour)

    p = sub.add_parser("train")
    model_args(p, adapter=False)
    p.add_argument("--mix", default="general", help="general | intent | reason | comma list of sources | none")
    p.add_argument("--train-file", default=None, help="your own JSONL: {state, options, answer, instruction?}")
    p.add_argument("--max-examples", type=int, default=None, help="stratified cap over sources")
    p.add_argument("--decontaminate", default=DEFAULT_DECONTAM, help="eval sets to decontaminate against, or none")
    p.add_argument("--out", required=True)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-targets", default="all-linear")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lam", type=float, default=1.0, help="KL anchor weight (paper default 1)")
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--seed", type=int, default=20260922)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("compare")
    p.add_argument("runs", nargs="+")
    p.add_argument("--tasks", default=None)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("export")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--push", default=None, help="Hub repo id, e.g. user/my-jev (private unless --public)")
    p.add_argument("--public", action="store_true")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("serve")
    model_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("general", help="GSM8K/IFEval/TriviaQA/LAMBADA/WikiText-2 via lm-eval")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--tasks", default="gsm8k,ifeval,triviaqa,lambada_openai,wikitext")
    p.add_argument("--limit", type=int, default=250)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_general)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
