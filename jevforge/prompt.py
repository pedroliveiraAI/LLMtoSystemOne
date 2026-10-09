"""Prompt rendering for LLM-as-Jev (arXiv 2610.02076v2, Sec. 3).

User turn: state, instruction, options as ``[1] .. [K]`` lines, then the fixed
answer instruction. Chat template with thinking disabled when the template
supports it; assistant turn prefilled with ``Best answer: [``.
"""

from __future__ import annotations

from dataclasses import dataclass

ANSWER_INSTRUCTION = "Answer only with its bracketed numeric identifier."
PREFILL = "Best answer: ["
DEFAULT_INSTRUCTION = "Choose the best option."


@dataclass
class RenderedPrompt:
    text: str          # full prompt text, ending with the prefill
    reply_start: str   # prompt text up to (not including) the prefill


def format_user_message(state: str, options: list[str], instruction: str | None = None) -> str:
    lines = [str(state).strip(), "", (instruction or DEFAULT_INSTRUCTION).strip(), "", "Options:"]
    lines += [f"[{i}] {str(o).strip()}" for i, o in enumerate(options, start=1)]
    lines += ["", ANSWER_INSTRUCTION]
    return "\n".join(lines)


def _template_supports_thinking_flag(tok) -> bool:
    tpl = getattr(tok, "chat_template", None)
    if isinstance(tpl, dict):
        tpl = " ".join(str(v) for v in tpl.values())
    return bool(tpl) and "enable_thinking" in tpl


def render_reply_start(tok, user_message: str, system: str | None = None) -> str:
    """Prompt text up to the start of the assistant reply."""
    if getattr(tok, "chat_template", None):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user_message}]
        kwargs = {"enable_thinking": False} if _template_supports_thinking_flag(tok) else {}
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    # Base models without a chat template: plain transcript format.
    head = f"{system}\n\n" if system else ""
    bos = tok.bos_token or ""
    return f"{bos}{head}User: {user_message}\nAssistant: "


def render(tok, state: str, options: list[str], instruction: str | None = None,
           system: str | None = None) -> RenderedPrompt:
    start = render_reply_start(tok, format_user_message(state, options, instruction), system)
    return RenderedPrompt(text=start + PREFILL, reply_start=start)


def suffix_text(k: int) -> str:
    """Candidate completion for option k (1-based). The closing bracket makes
    the set of suffixes prefix-free, so K is unbounded."""
    return f"{k}]"
