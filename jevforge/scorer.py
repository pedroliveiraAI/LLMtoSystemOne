"""Scoring candidates through the trie.

``tree`` mode: one forward pass over ``prompt + every trie token`` with a 4D
attention mask in which each trie token sees the prompt and its own ancestors
only (tree attention), and position ids by trie depth. All candidates share a
single prefill, as in the paper.

``cache`` mode: prefill the prompt once, replicate the cache K times, then
teacher-force the K suffixes as one right-padded batch. Works for recurrent and
hybrid architectures (Mamba, Gated DeltaNet in Qwen3.5) where a 4D tree mask
cannot express tree attention, and for multimodal-rope models.

``naive`` mode (the reference): one unpadded sequence per option, prompt + suffix,
with the model's own default positions. Exact for every architecture; the other
modes are checked against it, and it is the last-resort fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .trie import Encoded


@dataclass
class NodeLogProbs:
    node: torch.Tensor          # [N_nodes, V] log-probs of the next token at each trie node
    reply_start: torch.Tensor   # [V] log-probs at the reply-start position


def _device(model) -> torch.device:
    return next(model.parameters()).device


def _forward_logits(model, input_ids, attention_mask, position_ids, keep: torch.Tensor):
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                    logits_to_keep=keep, use_cache=False)
        return out.logits[0]
    except TypeError:
        # Older architectures without logits_to_keep: project only the rows we need.
        out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                    output_hidden_states=True, use_cache=False)
        return model.get_output_embeddings()(out.hidden_states[-1][0, keep])


def tree_forward(model, enc: Encoded) -> NodeLogProbs:
    dev = _device(model)
    nodes = enc.trie.nodes
    P, N = len(enc.prompt_ids), len(nodes)
    T = P + N - 1  # root has no token of its own
    ids = torch.tensor([enc.prompt_ids + [n.token for n in nodes[1:]]], device=dev)
    pos = list(range(P)) + [P - 1 + n.depth for n in nodes[1:]]
    allowed = torch.zeros(T, T, dtype=torch.bool, device=dev)
    allowed[:P, :P] = torch.tril(torch.ones(P, P, dtype=torch.bool, device=dev))
    for i in range(1, N):
        row = P + i - 1
        allowed[row, :P] = True
        j = i
        while j > 0:  # itself and every ancestor below the root
            allowed[row, P + j - 1] = True
            j = nodes[j].parent
    dtype = next(model.parameters()).dtype
    mask = torch.zeros(1, 1, T, T, dtype=dtype, device=dev)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    keep = torch.tensor([P - 1] + [P + i - 1 for i in range(1, N)] + [enc.reply_start_pos], device=dev)
    logits = _forward_logits(model, ids, mask, torch.tensor([pos], device=dev), keep).float()
    logp = torch.log_softmax(logits, dim=-1)
    return NodeLogProbs(node=logp[:N], reply_start=logp[N])


def reference_forward(model, enc: Encoded) -> NodeLogProbs:
    """Exact reference: each option scored on its own unpadded sequence."""
    dev = _device(model)
    nodes = enc.trie.nodes
    P = len(enc.prompt_ids)
    rows: list = [None] * len(nodes)
    rs = None
    for k, path in enumerate(enc.trie.paths):
        suf = enc.suffixes[k]
        ids = torch.tensor([enc.prompt_ids + suf], device=dev)
        keep = list(range(P - 1, P + len(suf))) + ([enc.reply_start_pos] if rs is None else [])
        logits = _forward_logits(model, ids, None, None, torch.tensor(keep, device=dev)).float()
        logp = torch.log_softmax(logits, -1)
        if rows[0] is None:
            rows[0] = logp[0]
        for t, v in enumerate(path):
            if rows[v] is None:
                rows[v] = logp[t + 1]
        if rs is None:
            rs = logp[-1]
    return NodeLogProbs(node=torch.stack(rows), reply_start=rs)


naive_forward = reference_forward  # backwards-compatible name


def _expand_cache(cache, n: int) -> None:
    """Replicate every per-batch tensor of every cache layer n times (KV, conv and
    recurrent states alike), so each candidate continues from the same prompt state."""
    def rep(t):
        return t.repeat_interleave(n, 0) if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == 1 else t
    for layer in getattr(cache, "layers", []):
        for name, val in list(vars(layer).items()):
            if torch.is_tensor(val):
                setattr(layer, name, rep(val))
            elif isinstance(val, dict):
                for i in list(val):
                    val[i] = rep(val[i])


def _freeze_state_writes(cache) -> None:
    """Recurrent layers write their new state into the cache in place (copy_). That
    breaks autograd for the suffix pass and is not needed (the state is never read
    again), so the suffix pass gets non-storing updates."""
    for layer in getattr(cache, "layers", []):
        if hasattr(layer, "update_recurrent_state"):
            layer.update_recurrent_state = lambda recurrent_states, state_idx=0, **kw: recurrent_states
        if hasattr(layer, "update_conv_state") and isinstance(getattr(layer, "conv_states", None), dict):
            prev = layer.conv_states

            def upd(conv_states, state_idx=0, _prev=prev, **kw):
                return torch.cat([_prev[state_idx], conv_states], dim=-1)
            layer.update_conv_state = upd


def cached_forward(model, enc: Encoded) -> NodeLogProbs:
    dev = _device(model)
    nodes = enc.trie.nodes
    P, K = len(enc.prompt_ids), len(enc.suffixes)
    out = model(input_ids=torch.tensor([enc.prompt_ids], device=dev), use_cache=True,
                logits_to_keep=torch.tensor([enc.reply_start_pos, P - 1], device=dev))
    head = torch.log_softmax(out.logits[0].float(), -1)
    cache = out.past_key_values
    _expand_cache(cache, K)
    _freeze_state_writes(cache)
    D = max(len(s) for s in enc.suffixes)
    pad = enc.suffixes[0][-1]
    ids = torch.full((K, D), pad, dtype=torch.long, device=dev)
    att = torch.ones(K, P + D, dtype=torch.long, device=dev)
    for k, s in enumerate(enc.suffixes):
        ids[k, :len(s)] = torch.tensor(s, device=dev)
        att[k, P + len(s):] = 0
    logits = model(input_ids=ids, attention_mask=att, past_key_values=cache, use_cache=True).logits.float()
    logp = torch.log_softmax(logits, -1)  # [K, D, V]: row k, position t = after consuming suffix token t
    rows: list = [None] * len(nodes)
    rows[0] = head[1]
    for k, path in enumerate(enc.trie.paths):
        for t, v in enumerate(path):
            if rows[v] is None:
                rows[v] = logp[k, t]
    return NodeLogProbs(node=torch.stack(rows), reply_start=head[0])


FORWARDS = {"tree": tree_forward, "cache": cached_forward, "naive": reference_forward}


def candidate_scores(nlp: NodeLogProbs, enc: Encoded) -> torch.Tensor:
    """s_k = sum_t log p(c_{k,t} | x, c_{k,<t})  ->  [K]"""
    nodes = enc.trie.nodes
    out = []
    for path in enc.trie.paths:
        s = nlp.node.new_zeros(())
        for v in path:
            s = s + nlp.node[nodes[v].parent, nodes[v].token]
        out.append(s)
    return torch.stack(out)


def legal_logmass(nlp: NodeLogProbs, enc: Encoded, v: int) -> torch.Tensor:
    """log m(v) = log sum_{w in legal(v)} p(w | x, v)"""
    legal = torch.tensor(enc.trie.nodes[v].legal, device=nlp.node.device)
    return torch.logsumexp(nlp.node[v, legal], dim=-1)


def tree_logq(nlp: NodeLogProbs, enc: Encoded) -> torch.Tensor:
    """log q(k|x) = sum_t [log p(c_t) - log m(c_<t)]  ->  [K]; sums to 1 over k."""
    nodes = enc.trie.nodes
    lm = {v: legal_logmass(nlp, enc, v) for v in enc.trie.internal()}
    out = []
    for path in enc.trie.paths:
        s = nlp.node.new_zeros(())
        for v in path:
            par = nodes[v].parent
            s = s + nlp.node[par, nodes[v].token] - lm[par]
        out.append(s)
    return torch.stack(out)
