"""Training objective of LLM-as-Jev (arXiv 2610.02076v2, Sec. 4).

    L = L_tree + lambda * (L_mass + L_out + L_pos)

* L_tree  = -log q(y|x), q locally renormalized over the legal tokens of each trie node
            on the gold path (only the gold path is evaluated).
* L_mass  = sum_v w_v KL_Bern(m0(v) || m(v))     legal vs illegal mass at trie nodes
* L_out   = sum_v w_v KL(p0_bar(.|v) || p_bar(.|v))  over illegal tokens, renormalized
* L_pos   = sum_u w_u KL(p0(.|u) || p(.|u))      full next-token KL outside the suffix
            (reply start, and right after the gold closing bracket)

Node weights: gold-path nodes share 0.5, three random off-path nodes share 0.5.
Position weights: 0.5 each. Anchored distributions keep the reference top-64 tokens plus
one tail bucket (a lower bound on the full KL).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .scorer import NodeLogProbs, tree_logq
from .trie import Encoded

TOPK = 64
N_OFF_PATH = 3


def topk_tail_kl(logp_ref: torch.Tensor, logp: torch.Tensor, k: int = TOPK) -> torch.Tensor:
    """KL(p_ref || p) over the reference top-k tokens plus one tail bucket. Inputs are
    log-probabilities (possibly -inf on excluded tokens) over the same support."""
    k = min(k, int(torch.isfinite(logp_ref).sum().item()))
    idx = torch.topk(logp_ref, k).indices
    keep = torch.zeros_like(logp_ref, dtype=torch.bool)
    keep[idx] = True
    kl = (logp_ref[idx].exp() * (logp_ref[idx] - logp[idx])).sum()
    rest_ref = logp_ref.masked_fill(keep, float("-inf"))
    if torch.isfinite(rest_ref).any():
        lt_ref = torch.logsumexp(rest_ref, -1)
        lt = torch.logsumexp(logp.masked_fill(keep, float("-inf")), -1)
        kl = kl + lt_ref.exp() * (lt_ref - lt)
    return kl.clamp(min=0.0)


def _split_mass(logp: torch.Tensor, legal: torch.Tensor):
    """(log legal mass, log illegal mass, illegal log-probs renormalized)."""
    is_legal = torch.zeros_like(logp, dtype=torch.bool)
    is_legal[legal] = True
    lm_legal = torch.logsumexp(logp[legal], -1)
    illegal = logp.masked_fill(is_legal, float("-inf"))
    lm_illegal = torch.logsumexp(illegal, -1)
    return lm_legal, lm_illegal, illegal - lm_illegal


def bernoulli_kl(lm0: torch.Tensor, lm0_c: torch.Tensor, lm: torch.Tensor, lm_c: torch.Tensor):
    """KL(Bern(m0) || Bern(m)) from log m and log(1-m) of both."""
    return (lm0.exp() * (lm0 - lm) + lm0_c.exp() * (lm0_c - lm_c)).clamp(min=0.0)


def anchor_nodes(enc: Encoded, gold: int, rng: random.Random) -> dict[int, float]:
    nodes = enc.trie.nodes
    gold_nodes = [nodes[v].parent for v in enc.trie.paths[gold]]  # decision nodes on the gold path
    others = [v for v in enc.trie.internal() if v not in set(gold_nodes)]
    off = rng.sample(others, min(N_OFF_PATH, len(others)))
    w_gold = 0.5 if off else 1.0
    weights = {v: w_gold / len(gold_nodes) for v in gold_nodes}
    for v in off:
        weights[v] = weights.get(v, 0.0) + 0.5 / len(off)
    return weights


@dataclass
class LossParts:
    total: torch.Tensor
    tree: float
    mass: float
    out: float
    pos: float
    root_logmass: float       # log m_theta(root), for the health check
    root_logmass_ref: float


def jev_loss(cur: NodeLogProbs, ref: NodeLogProbs, enc: Encoded, gold: int, lam: float = 1.0,
             rng: random.Random | None = None) -> LossParts:
    rng = rng or random.Random(0)
    l_tree = -tree_logq(cur, enc)[gold]
    zero = l_tree.new_zeros(())
    l_mass, l_out = zero, zero
    root_lm = root_lm_ref = None
    for v, w in anchor_nodes(enc, gold, rng).items():
        legal = torch.tensor(enc.trie.nodes[v].legal, device=cur.node.device)
        lm, lmc, ill = _split_mass(cur.node[v], legal)
        lm0, lm0c, ill0 = _split_mass(ref.node[v], legal)
        l_mass = l_mass + w * bernoulli_kl(lm0, lm0c, lm, lmc)
        l_out = l_out + w * topk_tail_kl(ill0, ill)
        if v == 0:
            root_lm, root_lm_ref = lm.item(), lm0.item()
    leaf = enc.trie.paths[gold][-1]
    l_pos = 0.5 * topk_tail_kl(ref.reply_start, cur.reply_start) + 0.5 * topk_tail_kl(ref.node[leaf], cur.node[leaf])
    total = l_tree + lam * (l_mass + l_out + l_pos)
    return LossParts(total=total, tree=l_tree.item(), mass=l_mass.item(), out=l_out.item(), pos=l_pos.item(),
                     root_logmass=root_lm, root_logmass_ref=root_lm_ref)
