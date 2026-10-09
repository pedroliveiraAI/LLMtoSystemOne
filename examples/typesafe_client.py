"""Cliente HTTP para o endpoint TypeSafe-compatível do jevforge.

1. Arranca o servidor (noutro terminal):
       JEVFORGE_API_KEY=um-segredo jevforge serve --model Qwen/Qwen3.5-4B --port 8765
2. Corre:
       python examples/typesafe_client.py

Só usa a biblioteca padrão. Para apontar ao TypeSafe real, muda JEV_URL e JEV_KEY.
"""

import json
import os
import urllib.request

JEV_URL = os.environ.get("JEV_URL", "http://127.0.0.1:8765")
JEV_KEY = os.environ.get("JEV_KEY", "um-segredo")


def decide(state, questions, timeout=600):
    """Envia um estado + perguntas tipadas; devolve {nome: resposta}."""
    body = json.dumps({"state": state, "questions": questions}).encode("utf-8")
    req = urllib.request.Request(
        f"{JEV_URL}/v1/systemone", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {JEV_KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())["answers"]


if __name__ == "__main__":
    ticket = ("Política: reembolsos exigem talão e compra há menos de 30 dias. "
              "O cliente comprou há 12 dias, não tem talão e pede o reembolso. Está irritado.")

    answers = decide(ticket, {
        # noul: sim/não -> probabilidade de "sim"
        "permitido": {"type": "noul",
                      "instructions": "A política permite o reembolso? Condições não provadas contam como não cumpridas.",
                      "criteria": {"true": "Todas as condições cumpridas", "false": "Falta uma condição"}},
        # choice: uma de várias opções -> label escolhido + probabilidades
        "acao": {"type": "choice",
                 "instructions": "Qual a próxima ação do agente?",
                 "criteria": {"refund": "Fazer o reembolso",
                              "decline": "Recusar e explicar a regra do talão",
                              "escalate": "Escalar para um supervisor"}},
        # score: nível ordinal -> índice do nível + probabilidades por nível
        "urgencia": {"type": "score",
                     "instructions": "Quão urgente é este ticket?",
                     "criteria": ["nada urgente", "algo urgente", "muito urgente"]},
    })

    p_yes = answers["permitido"]["noul"]
    print(f"permitido? {'sim' if p_yes >= 0.5 else 'não'} (p_sim={p_yes:.2f})")

    acao = answers["acao"]
    print(f"ação: {acao['choice']}  " + ", ".join(f"{k}={v:.2f}" for k, v in acao["probabilities"].items()))

    urg = answers["urgencia"]
    esperado = sum(int(k) * v for k, v in urg["probabilities"].items())  # valor esperado do nível
    print(f"urgência: nível {urg['score']} (esperado {esperado:.2f})")

    # A confiança é um número calibrado: usa-a para decidir quando pedir ajuda a um humano.
    if max(acao["probabilities"].values()) < 0.7:
        print("-> confiança baixa: enviar para revisão humana")
