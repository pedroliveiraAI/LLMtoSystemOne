import pytest

from jevforge.prompt import ANSWER_INSTRUCTION, PREFILL, render
from jevforge.trie import build_trie, encode


@pytest.mark.parametrize("tok_name", ["qwen_tok", "other_tok"])
def test_prefix_free_and_decodes_back(request, tok_name):
    tok = request.getfixturevalue(tok_name)
    for K in (1, 2, 9, 10, 77, 200):
        opts = [f"opt {i}" for i in range(K)]
        enc = encode(tok, render(tok, "state", opts), K)
        assert len(enc.suffixes) == K
        for i, a in enumerate(enc.suffixes):
            assert a, "empty suffix"
            for j, b in enumerate(enc.suffixes):
                if i != j:
                    assert b[:len(a)] != a, f"suffix {i + 1} is a prefix of {j + 1}"
        # prompt + suffix reconstructs the text of each candidate
        text = tok.decode(enc.prompt_ids + enc.suffixes[K - 1])
        assert text.rstrip().endswith(f"{K}]")


def test_qwen_trie_depth_bound(qwen_tok):
    import math
    for K in (9, 10, 100, 150):
        enc = encode(qwen_tok, render(qwen_tok, "s", ["o"] * K), K)
        assert enc.trie.depth <= math.floor(math.log10(K)) + 2


def test_prompt_format(qwen_tok):
    r = render(qwen_tok, "the state", ["alpha", "beta"], "Pick one.")
    assert r.text.endswith(PREFILL)
    assert "[1] alpha\n[2] beta" in r.text and ANSWER_INSTRUCTION in r.text
    assert "<think>\n\n</think>" in r.text  # thinking disabled on Qwen3


def test_build_trie_rejects_prefix():
    with pytest.raises(ValueError):
        build_trie([[1, 2], [1, 2, 3]])
