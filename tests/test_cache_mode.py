import torch

from jevforge.jev import Jev, select_mode
from jevforge.losses import jev_loss
from jevforge.prompt import render
from jevforge.scorer import cached_forward, candidate_scores, reference_forward
from jevforge.trie import encode


def _enc(tok, K):
    return encode(tok, render(tok, "Which is it?", [f"choice {i}" for i in range(K)]), K)


@torch.no_grad()
def test_cache_matches_reference_dense(tiny_model, qwen_tok):
    for K in (3, 12):
        enc = _enc(qwen_tok, K)
        a, b = cached_forward(tiny_model, enc), reference_forward(tiny_model, enc)
        assert torch.allclose(candidate_scores(a, enc), candidate_scores(b, enc), atol=1e-4)
        assert torch.allclose(a.reply_start, b.reply_start, atol=1e-4)


@torch.no_grad()
def test_cache_matches_reference_hybrid(tiny_hybrid, qwen_tok):
    for K in (3, 12):  # 12: suffixes of different lengths -> right padding through the recurrence
        enc = _enc(qwen_tok, K)
        a, b = cached_forward(tiny_hybrid, enc), reference_forward(tiny_hybrid, enc)
        assert torch.allclose(candidate_scores(a, enc), candidate_scores(b, enc), atol=1e-4)


def test_auto_mode_on_hybrid_picks_exact_mode(tiny_hybrid, qwen_tok):
    mode = select_mode(tiny_hybrid, _enc(qwen_tok, 4))
    assert mode in ("cache", "naive")
    jev = Jev(tiny_hybrid, qwen_tok)
    p = jev.decide("s", ["a", "b", "c"])
    assert abs(sum(p) - 1) < 1e-5 and jev.mode == mode


def test_gradient_through_cache_mode(tiny_hybrid, qwen_tok):
    enc = _enc(qwen_tok, 5)
    with torch.no_grad():
        ref = reference_forward(tiny_hybrid, enc)
    tiny_hybrid.zero_grad()
    cur = cached_forward(tiny_hybrid, enc)
    jev_loss(cur, ref, enc, gold=1).total.backward()
    grads = [p.grad for n, p in tiny_hybrid.named_parameters() if "linear_attn" in n and p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    tiny_hybrid.zero_grad()
