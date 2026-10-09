# jevforge

Transforma **qualquer LLM causal do Hugging Face** num **Jev**: um modelo de decisão que, dado um estado e K opções, devolve uma distribuição de probabilidade sobre as opções, sem gerar texto livre.

Implementa a receita de *LLM-as-Jev: LLMs Are Already Jev-Style Decision Models — When and How to Fine-Tune Them* (Li & Wagle, [arXiv 2610.02076v2](https://arxiv.org/abs/2610.02076v2)). O paper não publica código, por isso esta é uma implementação independente.

## Instalação

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"     # Linux/macOS: .venv/bin/pip
git clone https://github.com/fstandhartinger/jevbench third_party/jevbench
git -C third_party/jevbench checkout 1bcc55eb6c8cffde2306b3db03ede39b61c6152a
```

## Transformar o teu modelo num Jev

```bash
jevforge inspect  --model <hf-id>                                   # o modelo é compatível?
jevforge jevbench --model <hf-id> --out runs/base                   # baseline sem treino (JevBench-231)
jevforge train    --model <hf-id> --mix general --max-examples 3000 --out runs/meu-jev
jevforge jevbench --model <hf-id> --adapter runs/meu-jev --out runs/meu-jev
jevforge compare  runs/base runs/meu-jev                            # tabela vs Jev / paper
jevforge export   --model <hf-id> --adapter runs/meu-jev --out merged/meu-jev [--push user/meu-jev]
```

Em Python:

```python
from jevforge import Jev
jev = Jev.from_pretrained("Qwen/Qwen3-0.6B", adapter="runs/meu-jev")
jev.decide("Cliente comprou há 12 dias mas não tem talão. Reembolsar?",
           ["Reembolsar", "Recusar: falta o talão", "Escalar"],
           "Aplica a política: talão obrigatório e prazo de 30 dias.")
# -> [0.08, 0.86, 0.06]
```

Para usar dados teus, passa um JSONL com uma linha por exemplo:

```json
{"state": "...", "options": ["...", "..."], "answer": 1, "instruction": "..."}
```

Dá-se com `jevforge train --train-file meus.jsonl --mix none`. Também se pode usar `--mix general` para misturar com os dados do paper.

## Usar como substituto do TypeSafe (com ou sem treino)

`jevforge serve` expõe `POST /v1/systemone`, com o mesmo contrato da API TypeSafe: os mesmos tipos (`choice`, `noul`, `score`), o mesmo formato de pedido e de resposta, e autenticação Bearer. Os clientes TypeSafe só precisam de trocar o URL.

```bash
JEVFORGE_API_KEY=um-segredo jevforge serve --model Qwen/Qwen3.5-4B --port 8765   # [--adapter runs/meu-jev]
```

```bash
curl -X POST http://127.0.0.1:8765/v1/systemone -H "Authorization: Bearer um-segredo"   -H "Content-Type: application/json" -d '{
  "state": "Cliente comprou há 12 dias, sem talão, pede reembolso.",
  "questions": {
    "permitido": {"type": "noul",   "instructions": "A política permite o reembolso?"},
    "acao":      {"type": "choice", "instructions": "O que fazer?",
                  "criteria": {"refund": "Reembolsar", "decline": "Recusar", "escalate": "Escalar"}},
    "urgencia":  {"type": "score",  "instructions": "Quão urgente?", "criteria": ["baixa", "média", "alta"]}}}'
# -> {"answers": {"permitido": {"type": "noul", "noul": 0.12},
#                 "acao": {"type": "choice", "choice": "decline", "probabilities": {...}},
#                 "urgencia": {"type": "score", "score": 0, "probabilities": {"0": ..., "1": ..., "2": ...}}}, ...}
```

Em Python, sem servidor:

```python
from jevforge import Jev
from jevforge.typed import answer_all
jev = Jev.from_pretrained("Qwen/Qwen3.5-4B")
answer_all(jev, state, {"acao": {"type": "choice", "instructions": "...", "criteria": {...}}})
```

O Qwen3.5-4B sem treino faz 78.4% no JevBench-231, com ECE 0.041 e 100% de respostas válidas. Validei-o com o adapter oficial `typesafe` do harness apontado para este servidor: os resultados batem com a corrida in-process até 1e-7.

## Como funciona (paper, Sec. 3–4)

| Peça | Implementação |
|---|---|
| Prompt | Estado, instrução e opções `[1] … [K]`, terminando em "Answer only with its bracketed numeric identifier.". Chat template com thinking desligado e prefill `Best answer: [`. |
| Readout | s_k = Σ log p(sufixo `k]`) e P(k\|x) = softmax(s). O `]` torna os sufixos livres de prefixo, por isso K não tem limite. |
| Eficiência | Uma só forward pass sobre o prompt e todos os tokens da trie, com máscara de atenção em árvore. Há verificação automática contra a referência naive e fallback se o modelo não aceitar máscaras 4D. |
| Loss | L_tree = −log q(y\|x), com q renormalizado sobre os tokens legais de cada nó da trie. |
| Âncoras KL | L_mass (Bernoulli legal/ilegal), L_out (tokens ilegais renormalizados) e L_pos (início da resposta e depois do `]`). Usam top-64 tokens mais um balde de cauda, com λ=1. |
| LoRA | r=16, α=32, dropout 0, todas as camadas lineares, lr 1e-4, AdamW (wd 0.01), clip 1.0, warmup de 10% e decaimento até 10%, 1 época, batch 32. Para se a massa legal na raiz cair mais de 0.5 nats. |
| Dados | Mistura do paper (Table 4): CLINC150, MASSIVE, Bitext, CommonsenseQA, HellaSwag, ReClor, LogiQA 2.0, CosmosQA, PIQA, αNLI e ProofWriter. Decontaminação por 13-gramas e por frases iguais com 6 ou mais palavras, contra todos os benchmarks de avaliação. |

**Agnóstico ao modelo:**
- Os sufixos são tokenizados junto com o prompt, por isso funcionam com tokenizers dígito a dígito ou com agrupamento de dígitos.
- Os modelos sem chat template usam um formato simples.
- O LoRA usa `all-linear`.
- Está testado com Qwen3 e SmolLM2 (família Llama).

## Avaliação, com os mesmos benchmarks do paper

- **JevBench público (231 itens).** Corre no [harness oficial](https://github.com/fstandhartinger/jevbench), no commit `1bcc55e`, sem alterações (`jevforge/bench/jevbench_run.py`). Métricas: accuracy, ECE (10 bins), Brier, schema validity e consistência entre paráfrases. Os números são diretamente comparáveis com o Jev 1.13 (86.6%) e com o paper.
- **Externos**, com `jevforge eval`: BoolQ, MMLU, MMLU-Pro (800), ARC-Challenge, WinoGrande, SciQ e Banking77 (77 opções). Inclui a readout de letras (A/B/C) como baseline.
- **Comportamento**, com `jevforge behaviour`:
  - O que o modelo gera depois do `]`: para logo, ecoa a opção ou deriva. Compara-se com o modelo base.
  - Com `--conversation`: Dolly-15k e MT-Bench, medindo respostas que não terminam e loops de repetição em relação ao modelo base.
- **Capacidade geral**, com `jevforge general`: GSM8K, IFEval, TriviaQA, LAMBADA e WikiText-2, via lm-eval. Requer `pip install -e ".[general]"`.
- **Fora de âmbito**: MMBench e MMStar (multimodal).

## Estrutura

```
jevforge/  prompt.py trie.py scorer.py jev.py losses.py train.py evaluate.py compare.py export.py serve.py cli.py
           data/ (schema, hub, mix)   bench/ (jevbench_run)
tests/     trie, scorer (tree == naive), losses (âncoras = 0 na referência, loss à mão, gradientes)
scripts/   run_qwen3_0.6b.sh  — pipeline completo base → treino → avaliação → comparação
```
