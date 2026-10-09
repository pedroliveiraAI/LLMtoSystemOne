"""Tokenization of candidate suffixes and their prefix tree (trie).

Tokenizer-agnostic: each candidate is tokenized *together with* the prompt
(``prompt + "k]"``) and the shared prompt is the longest common token prefix.
This survives BPE merges across the boundary (e.g. a tokenizer that fuses
``[1`` into one token) and digit-grouping tokenizers (Phi-4, Llama 3), where a
whole number is one token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .prompt import RenderedPrompt, suffix_text


@dataclass
class Node:
    parent: int                 # -1 for the root
    token: int                  # edge token from parent (-1 for the root)
    depth: int                  # 0 for the root
    children: dict = field(default_factory=dict)  # token -> node index
    option: int | None = None   # 0-based option index when this node closes a candidate

    @property
    def legal(self) -> list[int]:
        return list(self.children.keys())


@dataclass
class Trie:
    nodes: list[Node]
    paths: list[list[int]]      # per option: node indices from root (exclusive) to leaf (inclusive)

    @property
    def depth(self) -> int:
        return max(n.depth for n in self.nodes)

    def internal(self) -> list[int]:
        return [i for i, n in enumerate(self.nodes) if n.children]


@dataclass
class Encoded:
    prompt_ids: list[int]       # shared prefix fed to the model
    reply_start_pos: int        # position whose logits predict the first reply token
    suffixes: list[list[int]]   # per option
    trie: Trie


def _lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def build_trie(suffixes: list[list[int]]) -> Trie:
    nodes = [Node(parent=-1, token=-1, depth=0)]
    paths = []
    for k, suf in enumerate(suffixes):
        cur, path = 0, []
        for t in suf:
            nxt = nodes[cur].children.get(t)
            if nxt is None:
                nxt = len(nodes)
                nodes.append(Node(parent=cur, token=t, depth=nodes[cur].depth + 1))
                nodes[cur].children[t] = nxt
            cur = nxt
            path.append(cur)
        if nodes[cur].option is not None or nodes[cur].children:
            raise ValueError(f"candidate suffixes are not prefix-free at option {k + 1}")
        nodes[cur].option = k
        paths.append(path)
    for n in nodes:
        if n.option is not None and n.children:
            raise ValueError("candidate suffixes are not prefix-free")
    return Trie(nodes=nodes, paths=paths)


def encode(tok, rendered: RenderedPrompt, n_options: int, suffix_fn=suffix_text) -> Encoded:
    """``suffix_fn(k)`` gives the completion text of 1-based option k."""
    if n_options < 1:
        raise ValueError("need at least one option")
    prompt_ids = _ids(tok, rendered.text)
    full = [_ids(tok, rendered.text + suffix_fn(k)) for k in range(1, n_options + 1)]
    shared = len(prompt_ids)
    for ids in full:
        shared = min(shared, _lcp(prompt_ids, ids), len(ids) - 1)
    if shared < 1:
        raise ValueError("prompt and candidates share no token prefix")
    suffixes = [ids[shared:] for ids in full]
    reply_start_pos = min(_lcp(_ids(tok, rendered.reply_start), prompt_ids), shared) - 1
    return Encoded(prompt_ids=prompt_ids[:shared], reply_start_pos=max(reply_start_pos, 0),
                   suffixes=suffixes, trie=build_trie(suffixes))
