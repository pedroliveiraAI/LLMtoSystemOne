"""Public API: turn any Hugging Face causal LM into a Jev-style decision model.

    jev = Jev.from_pretrained("Qwen/Qwen3-0.6B")
    jev.decide("Customer wants a refund ...", ["refund", "exchange", "escalate"])
    # -> [0.81, 0.07, 0.12]
"""

from __future__ import annotations

import string
import warnings

import torch

from .prompt import PREFILL, RenderedPrompt, format_user_message, render, render_reply_start
from .scorer import FORWARDS, candidate_scores, reference_forward
from .trie import encode

LETTER_INSTRUCTION = "Answer only with the letter of the best option."


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, dtype: str | None = None) -> torch.dtype:
    if dtype:
        return getattr(torch, dtype)
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_model_and_tokenizer(model_id: str, adapter: str | None = None, device: str | None = None,
                             dtype: str | None = None, revision: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(device)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    import transformers
    dkey = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **{dkey: pick_dtype(device, dtype)})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device).eval()
    return model, tok


@torch.no_grad()
def select_mode(model, enc) -> str:
    """tree (one pass, 4D tree mask) > cache (prefill once + replicated cache) > naive (exact,
    one sequence per option). A fast mode is kept only if it matches the exact reference."""
    ref = candidate_scores(reference_forward(model, enc), enc)
    tol = 1e-2 if next(model.parameters()).dtype == torch.float32 else 0.25
    for mode in ("tree", "cache"):
        try:
            diff = (candidate_scores(FORWARDS[mode](model, enc), enc) - ref).abs().max().item()
        except Exception as e:  # noqa: BLE001 - e.g. a model that rejects custom 4D masks
            warnings.warn(f"{mode} scoring unavailable ({type(e).__name__}: {str(e)[:120]})")
            continue
        if diff <= tol:
            return mode
        warnings.warn(f"{mode} scoring disagrees with the exact reference (max |ds|={diff:.3g})")
    return "naive"


class Jev:
    def __init__(self, model, tokenizer, mode: str = "auto", system: str | None = None):
        self.model = model
        self.tok = tokenizer
        self.mode = mode  # "auto" | "tree" | "cache" | "naive"
        self.system = system

    @classmethod
    def from_pretrained(cls, model_id: str, adapter: str | None = None, device: str | None = None,
                        dtype: str | None = None, revision: str | None = None, mode: str = "auto",
                        system: str | None = None) -> "Jev":
        model, tok = load_model_and_tokenizer(model_id, adapter, device, dtype, revision)
        return cls(model, tok, mode=mode, system=system)

    # -- internals -----------------------------------------------------------------------------
    def _node_logprobs(self, enc):
        if self.mode in FORWARDS:
            return FORWARDS[self.mode](self.model, enc)
        # auto: check the fast modes against the exact reference once, then commit to the first that agrees.
        self.mode = select_mode(self.model, enc)
        return FORWARDS[self.mode](self.model, enc)

    def encode(self, state: str, options: list[str], instruction: str | None = None):
        return encode(self.tok, render(self.tok, state, options, instruction, self.system), len(options))

    # -- public --------------------------------------------------------------------------------
    @torch.no_grad()
    def scores(self, state: str, options: list[str], instruction: str | None = None) -> torch.Tensor:
        enc = self.encode(state, options, instruction)
        return candidate_scores(self._node_logprobs(enc), enc)

    def decide(self, state: str, options: list[str], instruction: str | None = None) -> list[float]:
        """P(k | state) over the options, read from next-token probabilities."""
        return torch.softmax(self.scores(state, options, instruction), -1).tolist()

    @torch.no_grad()
    def decide_letters(self, state: str, options: list[str], instruction: str | None = None) -> list[float]:
        """Baseline readout from the paper: option letters A, B, C ... (K <= 26)."""
        if len(options) > 26:
            raise ValueError("letter readout supports at most 26 options")
        letters = string.ascii_uppercase
        msg = "\n".join([str(state).strip(), "", (instruction or "Choose the best option.").strip(), "",
                         "Options:"] + [f"{letters[i]}. {o}" for i, o in enumerate(options)]
                        + ["", LETTER_INSTRUCTION])
        start = render_reply_start(self.tok, msg, self.system)
        rendered = RenderedPrompt(text=start + "Best answer:", reply_start=start)
        enc = encode(self.tok, rendered, len(options), suffix_fn=lambda k: " " + letters[k - 1])
        return torch.softmax(candidate_scores(self._node_logprobs(enc), enc), -1).tolist()

    @torch.no_grad()
    def chat(self, message: str, max_new_tokens: int = 256) -> str:
        """Plain generation, to check the model still converses after training."""
        text = render_reply_start(self.tok, message, self.system)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


__all__ = ["Jev", "load_model_and_tokenizer", "pick_device", "pick_dtype", "format_user_message", "PREFILL"]
