#!/usr/bin/env bash
# Qwen3.5-4B (the paper's backbone) -> Jev. Hybrid Gated-DeltaNet model: jevforge auto-selects "cache" scoring.
# CPU budget: ~4.5h training (1000 examples) + ~1.5h evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONWARNINGS=ignore
JF=".venv/Scripts/jevforge"; [ -x "$JF" ] || JF="jevforge"
M="Qwen/Qwen3.5-4B"
BASE=runs/qwen3.5-4b-base
JEV=runs/qwen3.5-4b-jev
N_TRAIN=${N_TRAIN:-1000}
B77_LIMIT=${B77_LIMIT:-200}

# Each step is skipped when its result already exists, so the script resumes after an interruption.
[ -f $BASE/jevbench/summary.json ] || $JF jevbench --model $M --label "Qwen3.5-4B training-free (jevforge)" --out $BASE
[ -f $JEV/train_result.json ]      || $JF train    --model $M --mix general --max-examples $N_TRAIN --out $JEV --label "Qwen3.5-4B LoRA (jevforge)"
[ -f $JEV/jevbench/summary.json ]  || $JF jevbench --model $M --adapter $JEV --label "Qwen3.5-4B LoRA (jevforge)" --out $JEV
[ -f $BASE/eval/summary.json ]     || $JF eval     --model $M --tasks banking77 --readouts jev --limit $B77_LIMIT --out $BASE
[ -f $JEV/eval/summary.json ]      || $JF eval     --model $M --adapter $JEV --tasks banking77 --readouts jev --limit $B77_LIMIT --out $JEV
$JF compare $BASE $JEV | tee runs/comparison_qwen3.5-4b.md
