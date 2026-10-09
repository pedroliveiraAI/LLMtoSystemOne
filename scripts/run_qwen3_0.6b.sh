#!/usr/bin/env bash
# End-to-end: Qwen3-0.6B -> Jev, evaluated on the paper's benchmarks.
# CPU budget: ~3.5h training (3000 examples) + ~2-3h evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
JF=".venv/Scripts/jevforge"; [ -x "$JF" ] || JF="jevforge"
M="Qwen/Qwen3-0.6B"
BASE=runs/qwen3-0.6b-base
JEV=runs/qwen3-0.6b-jev
N_TRAIN=${N_TRAIN:-3000}
EVAL_LIMIT=${EVAL_LIMIT:-200}

$JF jevbench  --model $M --label "Qwen3-0.6B training-free (jevforge)" --out $BASE
$JF behaviour --model $M --out $BASE --post-bracket 231

$JF train     --model $M --mix general --max-examples $N_TRAIN --out $JEV --label "Qwen3-0.6B LoRA (jevforge)"

$JF jevbench  --model $M --adapter $JEV --label "Qwen3-0.6B LoRA (jevforge)" --out $JEV
$JF behaviour --model $M --adapter $JEV --out $JEV --post-bracket 231

$JF eval      --model $M --limit $EVAL_LIMIT --out $BASE
$JF eval      --model $M --adapter $JEV --limit $EVAL_LIMIT --out $JEV

$JF compare $BASE $JEV | tee runs/comparison.md
