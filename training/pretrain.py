"""GPT-OSS-Lite pre-training script."""
import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from models.transformer import GPTOSS, ModelConfig
from utils.checkpoint import CheckpointManager
from utils.distributed import device
from utils.logging import TrainingLogger
from utils.memory import assert_fits_in_available_gpu, estimate_model_memory_gb


def seed_everything(seed: int) -> None:
    """Seed all RNGs for reproducibility; call before model construction."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def make_warmup_cosine_lambda(warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.05):
    """Linear warmup → cosine decay → constant at min_lr_ratio."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step >= total_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


class PretrainDataset(Dataset):
    """Packed-token dataset: returns ``(input_ids, target_ids)`` windows."""

    _TORCH_SAVE_MAGIC_LEN = 8
    _LEGACY_DTYPES = (torch.long, torch.int32, torch.int64)

    def __init__(self, data_path: str, max_seq_len: int):
        self.max_seq_len = max_seq_len
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Pre-training data not found: {data_path}\nRun `python data/prepare_data.py` first.")
        self._load_manifest(data_path)
        if os.path.isdir(data_path):
            self._init_sharded(data_path)
        else:
            self._init_single(data_path)

    def _load_manifest(self, data_path: str) -> None:
        """Read EOS id from ``manifest.json`` if present; fall back silently."""
        manifest_path = Path(data_path) / "manifest.json"
        if manifest_path.exists():
            try:
                import json
                m = json.loads(manifest_path.read_text())
                self.eos_token_id = m.get("eos_token_id")
                self.vocab_size = m.get("vocab_size")
                self.total_tokens = m.get("total_tokens", 0)
                self.shard_count = m.get("shard_count", 0)
                self.dtype = m.get("dtype", "uint32")
                return
            except (json.JSONDecodeError, OSError):
                pass
        self.eos_token_id = None
        self.vocab_size = None
        self.total_tokens = 0
        self.shard_count = 0
        self.dtype = None

    def _detect_format(self, path: str) -> str:
        """Return ``"torch_save"`` or ``"raw_bytes"``."""
        with open(path, "rb") as f:
            magic = f.read(self._TORCH_SAVE_MAGIC_LEN)
        if magic[:2] == b"PK":
            return "torch_save"
        size = os.path.getsize(path)
        if size % 4 == 0:
            return "raw_bytes"
        return "torch_save"

    def _init_single(self, data_path: str) -> None:
        self.layout = "single"
        self.data = torch.load(data_path, weights_only=True, mmap=True)
        self._n_samples = max(1, (len(self.data) - 1) // self.max_seq_len)

    def _init_sharded(self, data_dir: str) -> None:
        shard_paths = sorted(Path(data_dir).glob("shard_*.bin"))
        if not shard_paths:
            raise FileNotFoundError(f"No `shard_*.bin` files in {data_dir}")
        self.layout = "sharded"
        self.shard_paths = [str(p) for p in shard_paths]
        self.shard_formats = [self._detect_format(p) for p in self.shard_paths]
        raw_dtype_map = {"uint32": torch.int32, "uint16": torch.int16, "uint8": torch.int8}
        self.raw_dtype = raw_dtype_map.get(self.dtype, torch.int32) if self.dtype else torch.int32
        self.shard_sizes = []
        self.shard_offsets = []
        running = 0
        for p, fmt in zip(self.shard_paths, self.shard_formats):
            n = self._size_in_tokens(p, fmt)
            self.shard_sizes.append(n)
            self.shard_offsets.append(running)
            running += n
        self._total = running
        self._n_samples = max(1, (self._total - 1) // self.max_seq_len)
        self._cache_shard_idx = -1
        self._cache_shard = None
        import bisect
        self._bisect = bisect

    def _size_in_tokens(self, path: str, fmt: str) -> int:
        """Return the number of tokens in a shard file."""
        size = os.path.getsize(path)
        if fmt == "torch_save":
            t = torch.load(path, weights_only=True, mmap=True)
            n = t.numel()
            del t
            return n
        return size // self.raw_dtype.itemsize

    def _load_shard(self, shard_idx: int):
        """Load a shard (mmap'd torch tensor). Caches the last-loaded shard."""
        if self._cache_shard_idx == shard_idx and self._cache_shard is not None:
            return self._cache_shard
        path = self.shard_paths[shard_idx]
        fmt = self.shard_formats[shard_idx]
        if fmt == "torch_save":
            t = torch.load(path, weights_only=True, mmap=True)
        else:
            t = torch.from_file(path, dtype=self.raw_dtype, shared=True, size=self.shard_sizes[shard_idx])
        self._cache_shard = t
        self._cache_shard_idx = shard_idx
        return t

    def __len__(self) -> int:
        return self._n_samples

    def _get_window_single(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.max_seq_len
        chunk = self.data[start: start + self.max_seq_len + 1]
        return chunk[:-1], chunk[1:]

    def _get_window_sharded(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.max_seq_len
        end = start + self.max_seq_len + 1
        last_offset = self.shard_offsets[-1]
        last_size = self.shard_sizes[-1]
        if start + self.max_seq_len < last_offset + last_size:
            shard_idx = self._bisect.bisect_right(self.shard_offsets, start) - 1
            shard_start = self.shard_offsets[shard_idx]
            shard_end = shard_start + self.shard_sizes[shard_idx]
            if end <= shard_end:
                shard = self._load_shard(shard_idx)
                local_start = start - shard_start
                chunk = shard[local_start: local_start + self.max_seq_len + 1]
                return chunk[:-1], chunk[1:]
        chunk = []
        pos = start
        i = 0
        while i < len(self.shard_paths) and pos < end:
            shard_start = self.shard_offsets[i]
            shard_end = shard_start + self.shard_sizes[i]
            if shard_end > pos:
                local_start = max(0, pos - shard_start)
                local_end = min(self.shard_sizes[i], end - shard_start)
                shard = self._load_shard(i)
                chunk.append(shard[local_start:local_end])
                pos = shard_start + local_end
            i += 1
        full = torch.cat(chunk)
        return full[:-1], full[1:]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.layout == "single":
            return self._get_window_single(idx)
        return self._get_window_sharded(idx)


def chunked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, chunk_size: int = 4096):
    """Memory-efficient cross-entropy over chunks (avoids materialising O(B*S*V))."""
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.view(-1)
    total_loss = torch.zeros((), device=flat_logits.device, dtype=flat_logits.dtype)
    n_total = flat_logits.size(0)
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk_loss = F.cross_entropy(flat_logits[start:end], flat_targets[start:end], reduction="sum")
        total_loss = total_loss + chunk_loss
    return total_loss / max(1, n_total)


def _set_hardware_perf_knobs() -> None:
    """Enable A100-specific performance knobs (TF32, cuDNN benchmark, cuBLASLt)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # benchmark_limit=0 forces exhaustive cuDNN algo search (default 10).
        # One-time cost at first step; converges to the fastest kernel for our
        # (B=8, T=4096) shape. Saves ~3-5% on A100.
        torch.backends.cudnn.benchmark_limit = 0
        # Route BF16 matmul through cuBLASLt (A100-optimized) instead of cuBLAS.
        # cuBLASLt has hand-tuned kernels for sm_80 that are 2-5% faster on
        # production shapes. Bit-exact (same numerics, different kernel choice).
        torch.backends.cuda.preferred_blas_library = "cublaslt"
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def main(
    config_path: str,
    max_steps: Optional[int] = None,
    dry_run: bool = False,
    seed: Optional[int] = None,
    resume_from: Optional[int] = None,
) -> None:
    if seed is not None:
        seed_everything(seed)
        print(f"[seed] Set torch/numpy/python RNGs to {seed}")

    _set_hardware_perf_knobs()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = ModelConfig(**cfg["model"])
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    if max_steps is not None:
        train_cfg["total_steps"] = max_steps

    accum = train_cfg.get("gradient_accumulation_steps", 1)
    if accum < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {accum}")
    micro_bs = train_cfg.get("micro_batch_size", 1)
    if micro_bs < 1:
        raise ValueError(f"micro_batch_size must be >= 1, got {micro_bs}")

    dev = device()
    model = GPTOSS(model_cfg).to(dev)
    n_params = model.num_parameters()
    n_active = model.num_active_parameters()
    print(f"[model] total params: {n_params / 1e6:.2f}M, active: {n_active / 1e6:.2f}M")

    compile_enabled = train_cfg.get("compile", False) and dev.type == "cuda"
    compile_mode = train_cfg.get("compile_mode", "max-autotune")
    if compile_enabled:
        try:
            model = torch.compile(model, mode=compile_mode, fullgraph=False)
            print(f"[compile] torch.compile enabled (mode={compile_mode})")
        except Exception as e:
            print(f"[compile] torch.compile failed ({e}); continuing without it")

    try:
        est = estimate_model_memory_gb(
            model,
            seq_len=model_cfg.max_seq_len,
            batch_size=micro_bs,
            grad_checkpoint=train_cfg.get("grad_checkpoint", True),
        )
        assert_fits_in_available_gpu(est)
    except RuntimeError as e:
        print(f"[memory] WARNING: {e}")

    no_decay = ["bias", "norm", "embed"]
    decay_params = [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)]
    no_decay_params = [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)]
    # eps=1e-6 (not 1e-8) for BF16 stability: BF16 has only 7 mantissa bits,
    # so 1e-8 underflows to denormal/zero in the 2nd-moment, silently stalling
    # late-stage convergence. DeepSeek-V3 and LLaMA-3 both use 1e-6.
    optim = AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg["lr"],
        betas=(train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.95)),
        eps=1e-6,
        foreach=True,
        fused=(dev.type == "cuda"),
    )
    sched = LambdaLR(optim, make_warmup_cosine_lambda(
        train_cfg["warmup_steps"],
        train_cfg["total_steps"],
        train_cfg.get("min_lr_ratio", 0.05),
    ))

    data_path = data_cfg["train_data_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            f"Run `python data/prepare_data.py` first."
        )
    ds = PretrainDataset(data_path, model_cfg.max_seq_len)
    num_workers = train_cfg.get("num_workers", 4)
    pin_memory = train_cfg.get("pin_memory", dev.type == "cuda")
    loader = DataLoader(
        ds,
        batch_size=micro_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    logger = TrainingLogger(log_interval=train_cfg.get("log_interval", 50), seq_len=model_cfg.max_seq_len)
    ckpt = CheckpointManager(train_cfg["save_dir"])

    nan_guard = train_cfg.get("nan_guard", True)
    nan_max_consec = train_cfg.get("nan_guard_max_consecutive", 5)
    nan_count = 0

    start_step = 0
    if resume_from is not None:
        meta = ckpt.load(model, step=resume_from, device=str(dev), optimizer=optim, scheduler=sched)
        start_step = meta["step"]
        print(f"[resume] Restored from step {start_step}")
        rng_path = ckpt.save_dir / f"rng_step_{resume_from}.pt"
        if rng_path.exists():
            rng_state = torch.load(rng_path, weights_only=False, map_location=dev)
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if dev.type == "cuda" and rng_state.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            print(f"[resume] Restored RNG state from step {start_step}")

    grad_ckpt = train_cfg.get("grad_checkpoint", True)
    if grad_ckpt:
        model.enable_gradient_checkpointing(every=train_cfg.get("grad_checkpoint_every", 3))

    model.train()
    step = start_step
    aux_alpha = train_cfg.get("aux_loss_alpha", 0.01)
    grad_clip = train_cfg["grad_clip"]

    pbar = tqdm(total=train_cfg["total_steps"], desc="pretrain", initial=step)
    optim.zero_grad(set_to_none=True)
    micro_step = 0

    log_interval = train_cfg.get("log_interval", 50)
    save_interval = train_cfg.get("save_interval", 2000)
    log_interval_safe = max(1, log_interval)
    save_interval_safe = max(1, save_interval)

    while step < train_cfg["total_steps"]:
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(dev, non_blocking=True)
            target_ids = target_ids.to(dev, non_blocking=True)

            with autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
                logits, aux_loss = model(input_ids)
                # chunk_size=8192 (was 4096): halves the number of CE kernel
                # launches from 8 to 4 at (B=8, T=4096). Saves ~20μs/step
                # at no cost (peak CE intermediate is 16GB, well under 80GB).
                ce = chunked_cross_entropy(logits, target_ids, chunk_size=8192)
                loss = (ce + aux_alpha * aux_loss) / accum

            if not torch.isfinite(loss):
                if nan_guard:
                    nan_count += 1
                    print(f"[nan-guard] step {step}: non-finite loss ({nan_count}/{nan_max_consec})")
                    optim.zero_grad(set_to_none=True)
                    micro_step = 0
                    if nan_count >= nan_max_consec:
                        print(f"[nan-guard] {nan_max_consec} consecutive NaNs — rolling back")
                        latest = ckpt.latest_step()
                        if latest is not None:
                            ckpt.load(model, step=latest, device=str(dev), optimizer=optim, scheduler=sched)
                            step = latest
                            nan_count = 0
                        else:
                            raise RuntimeError("NaN guard triggered with no checkpoint to roll back to.")
                    continue
                else:
                    raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

            nan_count = 0
            loss.backward()
            micro_step += 1

            is_accum_boundary = (micro_step % accum == 0)
            if is_accum_boundary:
                if grad_clip > 0:
                    try:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip, foreach=True
                        )
                    except TypeError:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                pbar.update(1)

                if step % log_interval_safe == 0:
                    lr = sched.get_last_lr()[0]
                    ce_val, aux_val = ce.item(), aux_loss.item()
                    logger.log(step, ce_val, metrics={"aux": aux_val}, lr=lr)

                if step > start_step and step % save_interval_safe == 0:
                    ckpt.save(model, optim, step, scheduler=sched, extra_meta={"aux_loss": aux_loss.item()})

                pbar.set_postfix(ce=f"{ce.item():.4f}", aux=f"{aux_loss.item():.4f}")

            if step >= train_cfg["total_steps"]:
                break

    final_meta = {"final": True, "seed": seed}
    ckpt.save(model, optim, step, scheduler=sched, extra_meta=final_meta)
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    rng_path = ckpt.save_dir / f"rng_step_{step}.pt"
    torch.save(rng_state, rng_path)
    logger.finish()
    print("[pretrain] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-OSS-Lite pre-training")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total_steps")
    parser.add_argument("--dry-run", action="store_true", help="Just check config + model builds, don't train")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--resume-from", type=int, default=None, help="Resume from checkpoint step")
    args = parser.parse_args()
    main(
        args.config,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        seed=args.seed,
        resume_from=args.resume_from,
    )