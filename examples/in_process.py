"""Usar o Jev diretamente em Python, sem servidor.

    python examples/in_process.py                      # Qwen3.5-4B (≈17 GB RAM em CPU)
    JEV_MODEL=Qwen/Qwen3-0.6B python examples/in_process.py   # versão rápida para testar
"""

import os

from jevforge import Jev
from jevforge.typed import answer_all

MODEL = os.environ.get("JEV_MODEL", "Qwen/Qwen3.5-4B")
ADAPTER = os.environ.get("JEV_ADAPTER")  # ex.: runs/qwen3.5-4b-jev depois de treinar

jev = Jev.from_pretrained(MODEL, adapter=ADAPTER)

# 1) Interface simples: estado + lista de opções -> probabilidades na mesma ordem
opcoes = ["Faturação", "Problema técnico", "Cancelar conta", "Outro"]
probs = jev.decide("Fui cobrado duas vezes este mês pela mesma subscrição.", opcoes,
                   "Para que equipa deve ir este pedido?")
for o, p in sorted(zip(opcoes, probs), key=lambda x: -x[1]):
    print(f"{p:6.1%}  {o}")

# 2) Interface tipada (igual ao TypeSafe): várias perguntas sobre o mesmo estado
pedido = {"cliente": "Ana", "plano": "pro", "dias_desde_compra": 12, "tem_talao": False,
          "mensagem": "Quero o meu dinheiro de volta."}
respostas = answer_all(jev, pedido, {
    "elegivel": {"type": "noul", "instructions": "Elegível para reembolso (talão e < 30 dias)?"},
    "tom": {"type": "choice", "instructions": "Tom da mensagem do cliente?",
            "criteria": ["calmo", "irritado", "neutro"]},
})
print("elegível:", f"{respostas['elegivel']['noul']:.2f}")
print("tom:", respostas["tom"]["choice"], respostas["tom"]["probabilities"])

# 3) Lote: decidir muitos itens (cada chamada é independente)
mensagens = ["A app crasha ao abrir", "Mudem o meu cartão de pagamento", "Adoro o produto!"]
for m in mensagens:
    p = jev.decide(m, opcoes, "Para que equipa deve ir este pedido?")
    print(f"{opcoes[p.index(max(p))]:<17} ({max(p):.0%})  <- {m}")
