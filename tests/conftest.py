"""Shared fixtures for GPT-OSS-Lite test suite."""
import shutil
import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def attn_small():
    """Small attention dims for fast CPU equivalence tests (<1s)."""
    return dict(B=2, T=256, H=4, D=32, window=128)


@pytest.fixture(scope="session")
def attn_tiny():
    """Even smaller dims for debugging / very fast shape tests."""
    return dict(B=1, T=64, H=2, D=16, window=32)


@pytest.fixture(scope="session")
def attn_large():
    """Production-scale dims — may be slow on CPU; mark slow."""
    return dict(B=2, T=4096, H=8, D=96, window=128)


@pytest.fixture(scope="session")
def yarn_cfg():
    """YaRN RoPE configuration matching the production 502M model."""
    return dict(
        head_dim=96,
        theta=100000,
        scale_factor=32,
        original_max_seq_len=4096,
        target_seq_len=131072,
        beta_fast=32,
        beta_slow=1,
    )


@pytest.fixture(scope="session")
def yarn_cfg_small():
    """YaRN RoPE configuration for fast CPU tests."""
    return dict(
        head_dim=32,
        theta=10000,
        scale_factor=4,
        original_max_seq_len=128,
        target_seq_len=512,
        beta_fast=4,
        beta_slow=1,
    )


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


def _make_attn_inputs(dims, seed=0):
    """Build random attention inputs (q, k, v) with a fixed seed."""
    B, T, H, D = dims["B"], dims["T"], dims["H"], dims["D"]
    torch.manual_seed(seed)
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    k = torch.randn(B, H, T, D, dtype=torch.float64)
    v = torch.randn(B, H, T, D, dtype=torch.float64)
    return q, k, v


@pytest.fixture
def attn_inputs_small(attn_small):
    return _make_attn_inputs(attn_small, seed=0)


@pytest.fixture
def attn_inputs_tiny(attn_tiny):
    return _make_attn_inputs(attn_tiny, seed=0)


@pytest.fixture(scope="session")
def model_cfg():
    """Full model config matching the 502M architecture (production)."""
    return {
        "vocab_size":              128000,
        "d_model":                 768,
        "n_layers":                12,
        "n_heads":                 8,
        "n_kv_heads":              4,
        "head_dim":                96,
        "ffn_dim":                 1536,
        "n_routed_experts":        8,
        "n_activated_experts":     2,
        "n_shared_experts":        1,
        "window_size":             128,
        "attention_pattern":       "alternating",
        "sink_bias":               True,
        "rope_theta":              100000,
        "yarn_scale_factor":       32,
        "yarn_original_max_seq_len": 4096,
        "yarn_target_seq_len":     131072,
        "yarn_beta_fast":          32,
        "yarn_beta_slow":          1,
        "yarn_mscale":             True,
        "yarn_prune_rope_global":  True,
        "max_seq_len":             4096,
        "eval_max_seq_len":        131072,
        "dtype":                   "bf16",
        "weight_tying":            True,
        "rms_norm_eps":            1e-5,
        "init_std":                0.02,
        "attn_impl":               "sdpa",
    }


@pytest.fixture(scope="session")
def small_cfg():
    """Small config for fast component tests (<2s on CPU)."""
    return {
        "vocab_size":              256,
        "d_model":                 64,
        "n_layers":                4,
        "n_heads":                 4,
        "n_kv_heads":              2,
        "head_dim":                16,
        "ffn_dim":                 128,
        "n_routed_experts":        4,
        "n_activated_experts":     2,
        "n_shared_experts":        1,
        "window_size":             32,
        "attention_pattern":       "alternating",
        "sink_bias":               True,
        "rope_theta":              10000,
        "yarn_scale_factor":       4,
        "yarn_original_max_seq_len": 128,
        "yarn_target_seq_len":     512,
        "yarn_beta_fast":          4,
        "yarn_beta_slow":          1,
        "yarn_mscale":             True,
        "yarn_prune_rope_global":  True,
        "max_seq_len":             128,
        "eval_max_seq_len":        512,
        "dtype":                   "bf16",
        "weight_tying":            True,
        "rms_norm_eps":            1e-5,
        "init_std":                0.02,
        "attn_impl":               "sdpa",
    }


@pytest.fixture()
def tmp_ckpt_dir():
    """Yield a temporary directory for checkpoint tests; clean up after."""
    tmp = Path(tempfile.mkdtemp(prefix="gptoss_ckpt_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture()
def tmp_data_dir():
    """Yield a temporary directory for data tests; clean up after."""
    tmp = Path(tempfile.mkdtemp(prefix="gptoss_data_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)