import pytest

from jevforge.jev import Jev
from jevforge.serve import make_app
from jevforge.typed import answer, question_to_options


def test_question_to_options_shapes():
    assert question_to_options({"type": "noul"})[1] == ["no", "yes"]
    assert question_to_options({"type": "score", "criteria": ["low", "mid", "high"]})[1] == ["0", "1", "2"]
    opts, labels = question_to_options({"type": "choice", "criteria": {"refund": "Give money back", "deny": None}})
    assert labels == ["refund", "deny"] and opts == ["refund: Give money back", "deny"]
    assert question_to_options({"type": "choice", "criteria": ["a", "b"]})[1] == ["a", "b"]
    with pytest.raises(ValueError):
        question_to_options({"type": "rank"})


def test_answer_shapes(tiny_model, qwen_tok):
    jev = Jev(tiny_model, qwen_tok)
    a = answer(jev, "s", {"type": "noul", "instructions": "ok?"})
    assert a["type"] == "noul" and 0 <= a["noul"] <= 1
    a = answer(jev, {"k": 1}, {"type": "choice", "instructions": "pick", "criteria": {"x": "X", "y": "Y"}})
    assert a["choice"] in ("x", "y") and abs(sum(a["probabilities"].values()) - 1) < 1e-5
    a = answer(jev, "s", {"type": "score", "instructions": "rate", "criteria": ["bad", "ok", "good"]})
    assert a["score"] in (0, 1, 2) and set(a["probabilities"]) == {"0", "1", "2"}


def test_systemone_endpoint(tiny_model, qwen_tok, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("JEVFORGE_API_KEY", "secret")
    client = TestClient(make_app(Jev(tiny_model, qwen_tok), "tiny"))
    body = {"state": "s", "questions": {"decision": {"type": "noul", "instructions": "ok?"},
                                        "route": {"type": "choice", "instructions": "who", "criteria": ["a", "b"]}}}
    assert client.post("/v1/systemone", json=body).status_code == 401
    r = client.post("/v1/systemone", json=body, headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200
    j = r.json()
    assert j["model"] == "tiny" and set(j["answers"]) == {"decision", "route"}
    bad = {"state": "s", "questions": {"d": {"type": "rank", "instructions": "x"}}}
    assert client.post("/v1/systemone", json=bad, headers={"Authorization": "Bearer secret"}).status_code == 422
