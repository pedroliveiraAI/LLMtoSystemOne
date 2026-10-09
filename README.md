# AnyLLMintoTypeSafe

Turn **any Hugging Face causal LLM** into a **Jev-style decision model**: you give it a state and a set of options, and it returns a calibrated probability for each option. It never generates free text. It also serves a **TypeSafe-compatible API** (`POST /v1/systemone`), so existing TypeSafe clients can point at a local, open-weights model by changing one URL.

The package is called `jevforge`. It is an independent implementation of *LLM-as-Jev: LLMs Are Already Jev-Style Decision Models — When and How to Fine-Tune Them* (Li & Wagle, [arXiv 2610.02076v2](https://arxiv.org/abs/2610.02076v2)). The paper ships no code.

```text
state + options ──► "[1] … [K]" prompt ──► LLM ──► P("1]"), P("2]"), … ──► {"choice": "decline", "probabilities": {...}}
```

- **No training needed to start.** Off the shelf, Qwen3.5-4B scores 78.4% on JevBench-231 with ECE 0.041, and 100% of its answers are valid.
- **Trainable.** Fine-tune with LoRA using the paper's tree-factorized loss and KL anchors, so the model gets better at deciding without forgetting how to chat.
- **Any architecture.** Plain transformers, models without a chat template, and hybrid recurrent models such as Qwen3.5 (Gated DeltaNet) are all supported. The scoring mode is chosen automatically and checked against an exact reference.
- **Comparable numbers.** Evaluation runs through the official [JevBench](https://github.com/fstandhartinger/jevbench) harness, unmodified.

## Results

These are the public JevBench items (231), scored by the official harness at commit `1bcc55e`. Accuracy is the argmax over each item's exact label set. ECE is top-label calibration with 10 bins; lower is better. All runs used CPU only.

| System | JevBench-231 | ECE | Banking77 (77-way) |
|---|---|---|---|
| **Qwen3.5-4B, training-free (this repo)** | **78.4%** (181/231) | **0.041** | — |
| **Qwen3-0.6B, training-free (this repo)** | 58.9% (136/231) | 0.237 | 24.0% |
| **Qwen3-0.6B + LoRA, 3k examples (this repo)** | 57.1% (132/231) | **0.124** | **40.5%** |
| *Cited:* paper, Qwen3.5-4B training-free / LoRA | 81.4% / 84.0% | 0.057 / 0.050 | 69.0% / 75.4% |
| *Cited:* paper, Qwen3-0.6B training-free / LoRA | 56.7% / 61.0% | 0.278 / 0.121 | 22.2% / 59.0% |
| *Cited:* Jev 1.13 (hosted TypeSafe) | 86.6% (200/231) | — | — |

- **Where the cited numbers come from.** The paper rows are from the paper itself. The Jev 1.13 row is from the [Open-Jev benchmark page](https://zefan-cai.github.io/open-jev/benchmarks/), which used an older harness commit. They cover the same items under slightly different conditions, so read gaps of 1–3 points with care.
- **The 0.6B LoRA run used about 3% of the paper's data.** It ran on CPU with 2,992 examples. The JevBench accuracy change is not significant (exact sign test, p = 0.57). Calibration halved, consistency across paraphrases rose from 75% to 86%, and Banking77 went from 24.0% to 40.5%.
- Per-item results, summaries and training logs are in [`runs/`](runs/).

## Install

```bash
git clone --recursive https://github.com/pedroliveiraAI/AnyLLMintoTypeSafe
cd AnyLLMintoTypeSafe
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,serve]"        # Linux/macOS: .venv/bin/pip
```

The `--recursive` flag also fetches the JevBench harness (a git submodule pinned at `1bcc55e`), which is needed only for `jevforge jevbench`. Requirements are Python ≥ 3.10, PyTorch, and transformers ≥ 4.51. Qwen3.5 needs transformers 5.x. A GPU is used automatically when present. On CPU, a 4B model needs about 17 GB of RAM in fp32.

## Bring your own LLM

Every command takes `--model`. It accepts anything `transformers.AutoModelForCausalLM` can load.

| Your model is… | Pass |
|---|---|
| on the Hugging Face Hub | `--model mistralai/Mistral-7B-Instruct-v0.3` |
| a folder on disk (downloaded, fine-tuned or merged) | `--model C:/models/my-llm` (a folder with `config.json` + weights + tokenizer) |
| private or gated on the Hub | run `huggingface-cli login` once, or set `HF_TOKEN=hf_…`, then `--model org/private-model` |
| pinned to a specific commit | `--model org/model --revision <commit-sha or tag>` |
| already fine-tuned into a Jev with this repo | `--model <base> --adapter runs/my-jev`, or the merged folder from `jevforge export` |

**Step 1: check the model with `inspect`.** It loads the model, renders the prompt, and prints three things:
- the trie depth for K = 4, 12 and 100 options;
- the **root legal mass**, which is how much probability the model already puts on answering with a bracketed number;
- the scoring mode it picked.

```bash
jevforge inspect --model <your-model>
```

```text
chat template: yes
K=4    prompt_tokens=61    trie_depth=2 nodes=9    root_legal_mass=0.9998 ...
K=100  prompt_tokens=917   trie_depth=4 nodes=201  root_legal_mass=0.9951 ...
scoring mode: tree
```

A root legal mass close to 1 means the model is usable as a Jev without training. A low value means the model does not follow the answer format yet. Expect that more from small or base (non-instruct) models. Fine-tuning targets exactly this: the tree loss puts probability on the legal identifiers.

**Step 2: measure it, then serve it.**

```bash
jevforge jevbench --model <your-model> --out runs/my-model     # JevBench-231: accuracy, ECE, Brier
jevforge serve    --model <your-model> --port 8765             # TypeSafe-compatible endpoint
```

**Options that apply to every command:**

| Flag | Default | Use |
|---|---|---|
| `--device` | `cuda` → `mps` → `cpu` | force a device, e.g. `--device cpu` |
| `--dtype` | `bfloat16` on CUDA, else `float32` | `--dtype float16` to halve memory on a GPU |
| `--mode` | `auto` | `tree`, `cache` or `naive`; `auto` picks the fastest mode that matches the exact reference |
| `--label` | the model id | the name shown in comparison tables and in the server's `model` field |

**What works:**
- Instruct and chat models with a chat template. This is the best case: the prompt goes through the model's own template, with thinking switched off when the template supports it.
- Base models without a chat template, which fall back to a plain `User: … / Assistant:` format.
- Any tokenizer, including digit-by-digit (Qwen) and digit-grouping ones (Llama 3, Phi-4).
- Attention-only and hybrid or recurrent architectures.

So far it has been tested with Qwen3, Qwen3.5 and SmolLM2 (Llama family). Encoder-only models (BERT-style) and pure image models are not supported.

**From Python**, either load by name or wrap a model you already have in memory (quantized, custom-loaded, and so on):

```python
from jevforge import Jev

jev = Jev.from_pretrained("C:/models/my-llm", device="cuda", dtype="bfloat16")

# or bring your own objects
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained("my-org/my-llm", device_map="auto",
                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True))
tok = AutoTokenizer.from_pretrained("my-org/my-llm")
jev = Jev(model.eval(), tok)                    # mode="auto" verifies the scoring mode on first use
```

## Use any LLM as a TypeSafe drop-in

```bash
JEVFORGE_API_KEY=my-secret jevforge serve --model Qwen/Qwen3.5-4B --port 8765
#                                         [--adapter runs/my-jev]   after fine-tuning
```

```bash
curl -X POST http://127.0.0.1:8765/v1/systemone \
  -H "Authorization: Bearer my-secret" -H "Content-Type: application/json" -d '{
  "state": "Policy: refunds need a receipt and purchase within 30 days. Customer bought 12 days ago, has no receipt, asks for a refund.",
  "questions": {
    "permitted": {"type": "noul",   "instructions": "Is the refund permitted?"},
    "action":    {"type": "choice", "instructions": "What should the agent do?",
                  "criteria": {"refund": "Issue the refund", "decline": "Decline and explain the receipt rule",
                               "escalate": "Escalate to a manager"}},
    "urgency":   {"type": "score",  "instructions": "How urgent is this ticket?",
                  "criteria": ["not urgent", "somewhat urgent", "very urgent"]}}}'
```

The response below is real, from Qwen3.5-4B without training:

```json
{"model": "Qwen/Qwen3.5-4B",
 "answers": {
   "permitted": {"type": "noul", "noul": 0.12},
   "action":    {"type": "choice", "choice": "decline",
                 "probabilities": {"refund": 0.015, "decline": 0.977, "escalate": 0.008}},
   "urgency":   {"type": "score", "score": 0, "probabilities": {"0": 0.43, "1": 0.40, "2": 0.17}}},
 "usage": {"input_tokens": 32, "output_tokens": 0}}
```

| Question type | `criteria` | Answer |
|---|---|---|
| `noul` (yes/no) | optional `{"true": …, "false": …}` | `noul` = P(yes) |
| `choice` | `{label: description}` or `[label, …]` | `choice` + `probabilities` per label |
| `score` (ordinal) | `[level_0, level_1, …]` | `score` (level index) + `probabilities` per level |

**Compatibility check.** The official JevBench `typesafe` adapter was pointed, unmodified, at this server. It returned 0 invalid answers on 24 JevBench items, 21 of them correct. Its probabilities match the in-process run to within 1e-7.

Ready-to-run clients are in [`examples/`](examples/). [`typesafe_client.py`](examples/typesafe_client.py) uses only the standard library, and [`in_process.py`](examples/in_process.py) runs without a server.

## Python API

```python
from jevforge import Jev
from jevforge.typed import answer_all

jev = Jev.from_pretrained("Qwen/Qwen3.5-4B")            # adapter="runs/my-jev" after training

# options in, probabilities out (same order)
jev.decide("I was charged twice this month.",
           ["Billing", "Technical issue", "Cancel account", "Other"],
           "Which team should handle this?")
# -> [0.60, 0.35, 0.01, 0.03]   (Qwen3.5-4B, training-free)

# TypeSafe-style typed questions; the state can be a dict
answer_all(jev, {"days_since_purchase": 12, "has_receipt": False}, {
    "eligible": {"type": "noul", "instructions": "Eligible for a refund (receipt and < 30 days)?"},
})
```

## Fine-tune your own Jev

```bash
jevforge inspect  --model <hf-id>                         # is this model usable? trie depth, legal mass, scoring mode
jevforge jevbench --model <hf-id> --out runs/base         # training-free baseline on JevBench-231
jevforge train    --model <hf-id> --mix general --max-examples 3000 --out runs/my-jev
jevforge jevbench --model <hf-id> --adapter runs/my-jev --out runs/my-jev
jevforge compare  runs/base runs/my-jev                   # table against the cited Jev / paper numbers
jevforge export   --model <hf-id> --adapter runs/my-jev --out merged/my-jev [--push user/my-jev]
```

To train on your own decisions, write one JSONL line per example and pass the file to `--train-file` (add `--mix none` to skip the paper's data mix):

```json
{"state": "…", "options": ["…", "…"], "answer": 1, "instruction": "…"}
```

[`scripts/run_qwen3_0.6b.sh`](scripts/run_qwen3_0.6b.sh) and [`scripts/run_qwen3.5_4b.sh`](scripts/run_qwen3.5_4b.sh) run the full pipeline (baseline → train → evaluate → compare). Each script resumes where it stopped.

## How it works

| Piece | What it does (paper Sec. 3–4) |
|---|---|
| Prompt | The prompt holds the state, the instruction, and options listed as `[1] … [K]`, ending with "Answer only with its bracketed numeric identifier." The chat template runs with thinking disabled, and the assistant turn is prefilled with `Best answer: [`. |
| Readout | Each option is scored as s_k = Σ log p(suffix `k]`), and P(k\|x) = softmax(s). The closing `]` makes the suffixes prefix-free (`1]` vs `12]`), so K is unbounded. |
| Loss | L_tree = −log q(y\|x), where q is renormalized over the legal tokens at each node of the candidate trie. Only the gold path is evaluated. |
| KL anchors | Three penalties tie the model to the frozen base: L_mass (legal vs. illegal mass at trie nodes), L_out (the distribution over illegal tokens) and L_pos (the full next-token distribution at the reply start and after `]`). Each uses the top 64 tokens plus a tail bucket, with λ = 1. |
| LoRA | r = 16, α = 32, dropout 0, all linear layers. AdamW at lr 1e-4 with weight decay 0.01, grad clip 1.0, 10% warmup then linear decay to 10%, 1 epoch, batch 32, options shuffled per example. The reference model is the same network with the adapter disabled, so no second copy is loaded. |
| Health check | Training stops if the root legal log-mass drops by more than 0.5 nats. |
| Data | The paper's mix (Table 4): CLINC150, MASSIVE, Bitext, CommonsenseQA, HellaSwag, ReClor, LogiQA 2.0, CosmosQA, PIQA, αNLI and ProofWriter. Each example is checked against every evaluation set and dropped if it shares a 13-gram or a sentence of 6 or more words. |

### Scoring modes (chosen automatically, verified against an exact reference)

| Mode | How | When |
|---|---|---|
| `tree` | One forward pass over the prompt plus every trie token, with a 4D tree-attention mask | Attention-only models (Qwen3, Llama, SmolLM2, …) |
| `cache` | Run the prompt once, replicate the cache K times (KV *and* recurrent/conv states), then teacher-force all suffixes as one batch | Hybrid or recurrent models, e.g. Qwen3.5 (24 Gated DeltaNet + 8 attention layers) |
| `naive` | Each option scored on its own unpadded sequence | The exact reference, and the last-resort fallback |

On Qwen3.5-4B, tree attention silently gives wrong scores: it was off by 3.3 log-units, because recurrent layers ignore attention masks. The automatic check rejects it and switches to `cache`, which matches the reference to 7.6e-6. During training, recurrent state writes are disabled on the suffix pass so that autograd works through the replicated cache.

## Evaluation

- **JevBench public-231**, with `jevforge jevbench`. The official harness runs unmodified (Runner, ledger, summarize). The only shim is a no-op `fcntl` on Windows. It reports accuracy, ECE, Brier, schema validity, ordinal MAE and paraphrase consistency.
- **External benchmarks**, with `jevforge eval`: BoolQ, MMLU, MMLU-Pro (800), ARC-Challenge, WinoGrande, SciQ and Banking77. A letter-logit readout (A/B/C…) is included as a baseline.
- **Behaviour**, with `jevforge behaviour`:
  - What the model generates after `]`: does it stop, echo the option, or drift? This is measured against the base model.
  - With `--conversation`, Dolly-15k and MT-Bench replies, checking for unterminated replies and repetition loops.
- **General capability**, with `jevforge general`: GSM8K, IFEval, TriviaQA, LAMBADA and WikiText-2, run through lm-eval.

## Layout

```
jevforge/
  prompt.py trie.py scorer.py     prompt contract, candidate trie, tree/cache/naive scoring
  jev.py typed.py serve.py        Jev API, TypeSafe-style typed questions, HTTP server
  losses.py train.py              LLM-as-Jev objective, LoRA training loop
  evaluate.py compare.py          metrics, behaviour checks, comparison tables
  data/                           HF dataset adapters, paper training mix, decontamination
  bench/jevbench_run.py           official JevBench harness driver
examples/                         HTTP client and in-process usage
scripts/                          end-to-end pipelines (Qwen3-0.6B, Qwen3.5-4B)
tests/                            21 tests: trie, scoring equivalence (incl. a hybrid model), losses, gradients, API
third_party/jevbench              official harness (git submodule, MIT)
runs/                             results reported above (no model weights)
```

## Limitations

- Multimodal decisions (MMBench, MMStar) are not implemented.
- Full fine-tuning with FSDP is not implemented; only LoRA is.
- The runs here are CPU-only and use small training subsets. The paper's full mix (more than 100k examples, plus its long-input data) needs a GPU.
- The "TypeSafe-compatible" claim is verified against the request and response contract that JevBench's `typesafe` adapter uses. Fields outside that contract are not implemented.

## Credits

- Method: Yinheng Li and Justin Wagle, [LLM-as-Jev (arXiv 2610.02076v2)](https://arxiv.org/abs/2610.02076v2).
- Benchmark and harness: [JevBench](https://github.com/fstandhartinger/jevbench) (MIT).
