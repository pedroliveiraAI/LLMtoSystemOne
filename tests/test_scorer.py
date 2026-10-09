import torch

from jevforge.jev import Jev
from jevforge.prompt import render
from jevforge.scorer import candidate_scores, naive_forward, tree_forward, tree_logq
from jevforge.trie import encode


def _enc(tok, K):
    return encode(tok, render(tok, "Which is it?", [f"choice {i}" for i in range(K)]), K)


@torch.no_grad()
def test_tree_matches_naive(tiny_model, qwen_tok):
    for K in (3, 12, 105):
        enc = _enc(qwen_tok, K)
        a, b = tree_forward(tiny_model, enc), naive_forward(tiny_model, enc)
        assert torch.allclose(candidate_scores(a, enc), candidate_scores(b, enc), atol=1e-4)
        assert torch.allclose(a.reply_start, b.reply_start, atol=1e-4)


@torch.no_grad()
def test_single_token_suffix_is_softmax_of_logits(tiny_model, qwen_tok):
    # K <= 9 on Qwen: suffix is "<digit>" "]", so s_k = log p(digit) + log p("]" | digit)
    enc = _enc(qwen_tok, 4)
    nlp = tree_forward(tiny_model, enc)
    s = candidate_scores(nlp, enc)
    for k, path in enumerate(enc.trie.paths):
        first, second = path
        expect = nlp.node[0, enc.trie.nodes[first].token] + nlp.node[first, enc.trie.nodes[second].token]
        assert torch.isclose(s[k], expect)


@torch.no_grad()
def test_tree_q_is_a_distribution(tiny_model, qwen_tok):
    for K in (2, 30):
        enc = _enc(qwen_tok, K)
        q = tree_logq(tree_forward(tiny_model, enc), enc).exp()
        assert torch.isclose(q.sum(), torch.tensor(1.0), atol=1e-5)


def test_jev_api(tiny_model, qwen_tok):
    jev = Jev(tiny_model, qwen_tok)
    p = jev.decide("state", ["a", "b", "c"])
    assert len(p) == 3 and abs(sum(p) - 1) < 1e-5
    assert jev.mode == "tree"  # auto-check accepted tree attention
    pl = jev.decide_letters("state", ["a", "b", "c"])
    assert len(pl) == 3 and abs(sum(pl) - 1) < 1e-5
