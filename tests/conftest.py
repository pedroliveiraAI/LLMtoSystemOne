import pytest
import torch


@pytest.fixture(scope="session")
def qwen_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.fixture(scope="session")
def other_tok():
    """A different tokenizer family with no chat template (sentencepiece, digit grouping)."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("BAAI/bge-m3")
    except Exception:  # noqa: BLE001
        pytest.skip("second tokenizer not available offline")


@pytest.fixture(scope="session")
def tiny_model(qwen_tok):
    """Random 2-layer Qwen3 with the real vocabulary: fast, exercises the real code path."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(qwen_tok), hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=4096)
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture(scope="session")
def tiny_hybrid(qwen_tok):
    """Random Qwen3.5-style hybrid: Gated DeltaNet (recurrent) layers + full attention."""
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(vocab_size=len(qwen_tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_key_head_dim=16,
                            linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                            max_position_embeddings=4096)
    return Qwen3_5ForCausalLM(cfg).eval()
