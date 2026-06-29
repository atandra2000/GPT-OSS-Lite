"""Inference tests: generation shape, mixed KV cache, passkey prompt construction."""
import pytest
import torch

from inference.long_context import (
    PASSKEY_PROMPT_TEMPLATE,
    PasskeyEvaluator,
    make_filler_text,
)
from inference.generate import MixedKVCache, generate
from models.transformer import GPTOSS, ModelConfig


# Mixed KV cache

def test_kv_cache_initial_empty():
    """A fresh MixedKVCache must have length 0."""
    cache = MixedKVCache()
    assert len(cache) == 0


def test_kv_cache_append_windowed():
    """Windowed layers must keep only the last `window` tokens."""
    cache = MixedKVCache()
    for i in range(5):
        k = torch.zeros(1, 2, 1, 8)
        v = torch.zeros(1, 2, 1, 8)
        cache.append(layer_idx=0, k_rot=k, v=v, is_windowed=True, window=3)
    assert cache.windowed_kv[0][0].size(2) == 3
    assert cache.windowed_kv[0][1].size(2) == 3
    assert cache.seq_len(0, is_windowed=True) == 3


def test_kv_cache_append_global():
    """Global layers must accumulate all tokens (no truncation)."""
    cache = MixedKVCache()
    for i in range(5):
        k = torch.zeros(1, 2, 1, 8)
        v = torch.zeros(1, 2, 1, 8)
        cache.append(layer_idx=0, k_rot=k, v=v, is_windowed=False, window=3)
    assert cache.seq_len(0, is_windowed=False) == 5
    k_view, v_view = cache.get(0, is_windowed=False)
    assert k_view.size(2) == 5
    assert v_view.size(2) == 5


def test_kv_cache_reset():
    """reset() must empty both windowed and global caches."""
    cache = MixedKVCache()
    cache.append(0, torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8), True, 8)
    cache.append(0, torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8), False, 8)
    assert len(cache) > 0
    cache.reset()
    assert len(cache) == 0


def test_kv_cache_windowed_preserves_order_after_rollover():
    """The ring buffer must preserve the temporal order of the last `window` keys.

    Append 7 distinct (B=1, H=1, T=1, D=2) keys to a window=3 cache. After
    rollover, the cached tensor should contain keys [5, 6, 7] in that order.
    """
    cache = MixedKVCache()
    for i in range(7):
        k = torch.full((1, 1, 1, 2), float(i + 1))
        v = torch.zeros(1, 1, 1, 2)
        cache.append(layer_idx=0, k_rot=k, v=v, is_windowed=True, window=3)
    cached_k, _ = cache.get(0, is_windowed=True)
    expected = torch.tensor([5.0, 6.0, 7.0]).view(1, 1, 3, 1)
    assert torch.allclose(cached_k[..., 0], expected.expand(1, 1, 3, 1)[..., 0]), \
        f"Ring buffer order wrong: {cached_k[..., 0]}"


def test_kv_cache_global_preserves_full_order():
    """The global cache must contain all tokens in insertion order."""
    cache = MixedKVCache()
    for i in range(7):
        k = torch.full((1, 1, 1, 1), float(i + 1))
        v = torch.full((1, 1, 1, 1), float(i + 1))
        cache.append(layer_idx=0, k_rot=k, v=v, is_windowed=False, window=3)
    cached_k, cached_v = cache.get(0, is_windowed=False)
    expected = torch.arange(1, 8, dtype=torch.float32).view(1, 1, 7, 1)
    assert torch.allclose(cached_k, expected), f"Global K order: {cached_k}"
    assert torch.allclose(cached_v, expected), f"Global V order: {cached_v}"


def test_kv_cache_seq_len_helper():
    """seq_len() must report 0 for an empty cache and the active length otherwise."""
    cache = MixedKVCache()
    assert cache.seq_len(0, is_windowed=True) == 0
    assert cache.seq_len(0, is_windowed=False) == 0
    for i in range(5):
        k = torch.zeros(1, 2, 1, 8)
        v = torch.zeros(1, 2, 1, 8)
        cache.append(0, k, v, is_windowed=True, window=3)
    assert cache.seq_len(0, is_windowed=True) == 3  # capped at window


# generate()

def test_generate_shape():
    """generate() must produce input + max_new_tokens length output."""
    cfg = ModelConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        ffn_dim=64,
        n_routed_experts=2,
        n_activated_experts=1,
        n_shared_experts=1,
        window_size=8,
        max_seq_len=32,
        rope_theta=10000,
        yarn_scale_factor=2,
        yarn_original_max_seq_len=32,
        yarn_target_seq_len=64,
        yarn_beta_fast=2,
        yarn_beta_slow=1,
        yarn_prune_rope_global=False,
    )
    model = GPTOSS(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    output = generate(model, input_ids, max_new_tokens=8, temperature=0.0)
    assert output.shape == (1, 4 + 8)


def test_generate_no_crash_greedy():
    """Greedy generation (temperature=0) must not crash."""
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        ffn_dim=64,
        n_routed_experts=2,
        n_activated_experts=1,
        n_shared_experts=1,
        window_size=8,
        max_seq_len=16,
        rope_theta=10000,
        yarn_scale_factor=2,
        yarn_original_max_seq_len=16,
        yarn_target_seq_len=32,
        yarn_beta_fast=2,
        yarn_beta_slow=1,
        yarn_prune_rope_global=False,
    )
    model = GPTOSS(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = generate(model, input_ids, max_new_tokens=4, temperature=0.0, top_p=1.0)
    assert out.size(1) == 4 + 4


def test_generate_use_cache_false_matches_cache_for_one_token():
    """use_cache=False with max_new_tokens=1 must match use_cache=True for the same input."""
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        ffn_dim=64,
        n_routed_experts=2,
        n_activated_experts=1,
        n_shared_experts=1,
        window_size=8,
        max_seq_len=16,
        rope_theta=10000,
        yarn_scale_factor=2,
        yarn_original_max_seq_len=16,
        yarn_target_seq_len=32,
        yarn_beta_fast=2,
        yarn_beta_slow=1,
        yarn_prune_rope_global=False,
    )
    model = GPTOSS(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    out_cache = generate(model, input_ids, max_new_tokens=1, temperature=0.0, use_cache=True)
    out_no_cache = generate(model, input_ids, max_new_tokens=1, temperature=0.0, use_cache=False)
    assert out_cache.shape == out_no_cache.shape == (1, 9)
    assert torch.equal(out_cache, out_no_cache), (
        "use_cache=False must reproduce the same logits as use_cache=True when "
        "the no-cache path replays the full history each step."
    )


# Passkey evaluator (prompt construction only, no model eval)

class _StubTokenizer:
    """Whitespace tokenizer for prompt-structure tests."""
    def encode(self, s): return s.split()
    def decode(self, ids): return " ".join(str(i) for i in ids)


def test_make_filler_text_length():
    """make_filler_text must produce roughly target_tokens of text."""
    text = make_filler_text(target_tokens=128, seed=0)
    n_tokens = len(text.split())
    assert 100 <= n_tokens <= 200


def test_passkey_prompt_contains_passkey():
    """The prompt must contain the passkey at the specified position."""
    from models.transformer import GPTOSS, ModelConfig
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        ffn_dim=64,
        n_routed_experts=2,
        n_activated_experts=1,
        n_shared_experts=1,
        window_size=8,
        max_seq_len=16,
        rope_theta=10000,
        yarn_scale_factor=2,
        yarn_original_max_seq_len=16,
        yarn_target_seq_len=32,
        yarn_beta_fast=2,
        yarn_beta_slow=1,
        yarn_prune_rope_global=False,
    )
    model = GPTOSS(cfg)
    model.eval()
    evaluator = PasskeyEvaluator(model, _StubTokenizer())
    prompt = evaluator.build_prompt("12345", context_length=64, passkey_position="middle", seed=0)
    assert "12345" in prompt
    retrieval_idx = prompt.find("important info")
    passkey_idx = prompt.find("12345")
    assert passkey_idx < retrieval_idx


def test_passkey_position_start_middle_end():
    """Passkey position must be honored."""
    from models.transformer import GPTOSS, ModelConfig
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        ffn_dim=64,
        n_routed_experts=2,
        n_activated_experts=1,
        n_shared_experts=1,
        window_size=8,
        max_seq_len=16,
        rope_theta=10000,
        yarn_scale_factor=2,
        yarn_original_max_seq_len=16,
        yarn_target_seq_len=32,
        yarn_beta_fast=2,
        yarn_beta_slow=1,
        yarn_prune_rope_global=False,
    )
    model = GPTOSS(cfg)
    model.eval()
    evaluator = PasskeyEvaluator(model, _StubTokenizer())
    p_start = evaluator.build_prompt("99999", 64, "start", seed=0)
    p_middle = evaluator.build_prompt("99999", 64, "middle", seed=0)
    p_end = evaluator.build_prompt("99999", 64, "end", seed=0)
    assert p_start != p_middle
    assert p_middle != p_end
    assert p_start != p_end


def test_extract_passkey_from_output():
    """extract_passkey_from_output must find a 5-digit number."""
    from models.transformer import GPTOSS, ModelConfig
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        ffn_dim=64,
        n_routed_experts=2,
        n_activated_experts=1,
        n_shared_experts=1,
        window_size=8,
        max_seq_len=16,
        rope_theta=10000,
        yarn_scale_factor=2,
        yarn_original_max_seq_len=16,
        yarn_target_seq_len=32,
        yarn_beta_fast=2,
        yarn_beta_slow=1,
        yarn_prune_rope_global=False,
    )
    model = GPTOSS(cfg)
    model.eval()
    evaluator = PasskeyEvaluator(model, _StubTokenizer())
    assert evaluator.extract_passkey_from_output("The passkey is 12345") == "12345"
    assert evaluator.extract_passkey_from_output("I think it's 99999 or 12345") == "99999"
    assert evaluator.extract_passkey_from_output("no number here") is None
    assert evaluator.extract_passkey_from_output("1234 is the answer") is None