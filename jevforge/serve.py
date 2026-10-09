"""HTTP API.

* ``POST /v1/systemone`` - TypeSafe-compatible typed decisions:
  ``{"state": ..., "model"?: str, "questions": {name: {type, instructions, criteria}}}``
  -> ``{"model": str, "answers": {name: answer}, "usage": {...}}`` (see ``jevforge.typed``).
  If ``JEVFORGE_API_KEY`` is set, requests must send ``Authorization: Bearer <key>``.
* ``POST /decide`` - plain options: ``{state, options, instruction?}`` -> ``{probs, choice, index}``.
"""


import os
import time

from .typed import answer_all, state_text


def make_app(jev, model_name: str = "jevforge"):
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel

    class Req(BaseModel):
        state: str
        options: list[str]
        instruction: str | None = None

    class TypedReq(BaseModel):
        state: str | dict | list
        questions: dict[str, dict]
        model: str | None = None

    app = FastAPI(title="jevforge")
    api_key = os.environ.get("JEVFORGE_API_KEY")

    @app.post("/v1/systemone")
    def systemone(req: TypedReq, authorization: str | None = Header(default=None)):
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")
        t0 = time.perf_counter()
        try:
            answers = answer_all(jev, req.state, req.questions)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        n_in = len(jev.tok(state_text(req.state), add_special_tokens=False)["input_ids"])
        return {"model": model_name, "answers": answers,
                "usage": {"input_tokens": n_in, "output_tokens": 0},
                "latency_s": round(time.perf_counter() - t0, 3)}

    @app.post("/decide")
    def decide(req: Req):
        p = jev.decide(req.state, req.options, req.instruction)
        k = max(range(len(p)), key=p.__getitem__)
        return {"probs": p, "index": k, "choice": req.options[k]}

    @app.get("/health")
    def health():
        return {"ok": True, "model": model_name, "scoring_mode": jev.mode}

    return app


def serve(jev, host: str = "127.0.0.1", port: int = 8000, model_name: str = "jevforge"):
    import uvicorn
    uvicorn.run(make_app(jev, model_name), host=host, port=port)
