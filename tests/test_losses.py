import random

import torch

from jevforge.losses import anchor_nodes, jev_loss, topk_tail_kl
from jevforge.prompt import render
from jevforge.scorer import NodeLogProbs, tree_forward
from jevforge.trie import encode


def _enc(tok, K):
    return encode(tok, render(tok, "s", [f"o{i}" for i in range(K)]), K)


def test_anchors_vanish_at_reference(tiny_model, qwen_tok):
    enc = _enc(qwen_tok, 25)
    with torch.no_grad():
        ref = tree_forward(tiny_model, enc)
    parts = jev_loss(ref, ref, enc, gold=17)
    assert abs(parts.mass) < 1e-6 and abs(parts.out) < 1e-6 and abs(parts.pos) < 1e-6
    assert abs(parts.total.item() - parts.tree) < 1e-5


def test_tree_loss_by_hand(qwen_tok):
    enc = _enc(qwen_tok, 3)  # depth 2: root -> digit -> "]"
    V = len(qwen_tok)
    torch.manual_seed(0)
    node = torch.log_softmax(torch.randn(len(enc.trie.nodes), V), -1)
    nlp = NodeLogProbs(node=node, reply_start=node[0])
    gold = 1
    d, b = enc.trie.paths[gold]
    root_legal = enc.trie.nodes[0].legal
    expect = -((node[0, enc.trie.nodes[d].token] - torch.logsumexp(node[0, root_legal], -1))
               + (node[d, enc.trie.nodes[b].token] - torch.logsumexp(node[d, enc.trie.nodes[d].legal], -1)))
    parts = jev_loss(nlp, nlp, enc, gold)
    assert abs(parts.tree - expect.item()) < 1e-5


def test_topk_tail_kl_is_lower_bound_of_full_kl():
    torch.manual_seed(1)
    p0, p = torch.log_softmax(torch.randn(500), -1), torch.log_softmax(torch.randn(500), -1)
    full = (p0.exp() * (p0 - p)).sum()
    approx = topk_tail_kl(p0, p, k=64)
    assert 0 <= approx <= full + 1e-6


def test_anchor_weights():
    class E:  # minimal stand-in
        pass
    from jevforge.trie import build_trie
    e = E()
    e.trie = build_trie([[i % 10, i // 10, 99] if i >= 10 else [i, 99] for i in range(1, 40)])
    w = anchor_nodes(e, 25, random.Random(0))
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_gradient_flows(tiny_model, qwen_tok):
    enc = _enc(qwen_tok, 5)
    with torch.no_grad():
        ref = tree_forward(tiny_model, enc)
    tiny_model.zero_grad()
    jev_loss(tree_forward(tiny_model, enc), ref, enc, gold=2).total.backward()
    g = tiny_model.model.layers[0].self_attn.q_proj.weight.grad
    assert g is not None and g.abs().sum() > 0
    tiny_model.zero_grad()
