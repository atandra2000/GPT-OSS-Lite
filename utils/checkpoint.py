"""Atomic safetensors checkpoint manager with shared-tensor dedup and step discovery."""
import json, logging, os, tempfile
from pathlib import Path
from typing import Optional
import torch
from safetensors.torch import save_file, load_file

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Save/load model checkpoints. Files: model_step_N.safetensors, optim_step_N.pt, meta_step_N.json."""
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int,
             scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
             extra_meta: Optional[dict] = None) -> None:
        self._atomic_save_safetensors(model.state_dict(), self.save_dir / f"model_step_{step}.safetensors")
        self._atomic_save_torch(optimizer.state_dict(), self.save_dir / f"optim_step_{step}.pt")
        if scheduler is not None:
            self._atomic_save_torch(scheduler.state_dict(), self.save_dir / f"sched_step_{step}.pt")
        meta: dict = {"step": step}
        if extra_meta:
            meta.update(extra_meta)
        self._atomic_save_json(meta, self.save_dir / f"meta_step_{step}.json")
        logger.info("[checkpoint] saved step %d → %s", step, self.save_dir)

    def load(self, model: torch.nn.Module, step: int, device: str = "cuda",
             optimizer: Optional[torch.optim.Optimizer] = None, 
             scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
             strict: bool = True) -> dict:
        weight_path = self.save_dir / f"model_step_{step}.safetensors"
        if not weight_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {weight_path}\nAvailable steps: {self._list_steps()}")
        weights = load_file(str(weight_path), device=device)
        missing, unexpected = model.load_state_dict(weights, strict=False)
        if missing:
            msg = f"[checkpoint] {len(missing)} missing key(s): {missing[:5]}{'…' if len(missing) > 5 else ''}"
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)
        if unexpected:
            msg = f"[checkpoint] {len(unexpected)} unexpected key(s): {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}"
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)
        if optimizer is not None:
            optim_path = self.save_dir / f"optim_step_{step}.pt"
            if optim_path.exists():
                optimizer.load_state_dict(torch.load(optim_path, map_location=device, weights_only=True))
            else:
                logger.warning("[checkpoint] no optimiser state at %s — optimizer will start from scratch", optim_path)
        if scheduler is not None:
            sched_path = self.save_dir / f"sched_step_{step}.pt"
            if sched_path.exists():
                scheduler.load_state_dict(torch.load(sched_path, map_location=device, weights_only=True))
            else:
                logger.warning("[checkpoint] no scheduler state at %s — scheduler will start from scratch", sched_path)
        meta_path = self.save_dir / f"meta_step_{step}.json"
        meta: dict = json.load(open(meta_path)) if meta_path.exists() else {"step": step}
        logger.info("[checkpoint] loaded step %d from %s", step, self.save_dir)
        return meta

    def latest_step(self) -> Optional[int]:
        steps = self._list_steps()
        return next((s for s in sorted(steps, reverse=True) if self._checkpoint_complete(s)), None)

    def list_checkpoints(self) -> list:
        return sorted(s for s in self._list_steps() if self._checkpoint_complete(s))

    def delete_checkpoint(self, step: int) -> None:
        for pattern in [f"model_step_{step}.safetensors", f"optim_step_{step}.pt", f"sched_step_{step}.pt", f"meta_step_{step}.json"]:
            p = self.save_dir / pattern
            if p.exists():
                p.unlink()
        logger.info("[checkpoint] deleted step %d", step)

    def keep_last_n(self, n: int) -> None:
        complete = self.list_checkpoints()
        for step in complete[:-n]:
            self.delete_checkpoint(step)

    def _atomic_save_safetensors(self, state: dict, path: Path) -> None:
        # dedup shared-storage tensors so safetensors doesn't error on duplicate data_ptr
        seen_ptrs: set = set()
        deduped: dict = {}
        for k, v in state.items():
            ptr = v.data_ptr()
            if ptr in seen_ptrs:
                deduped[k] = v.contiguous().clone()
            else:
                seen_ptrs.add(ptr)
                deduped[k] = v.contiguous()
        self._atomic_write(path, lambda tmp: save_file(deduped, tmp), suffix=".safetensors.tmp")

    def _atomic_save_torch(self, obj, path: Path) -> None:
        self._atomic_write(path, lambda tmp: torch.save(obj, tmp), suffix=".pt.tmp")

    def _atomic_save_json(self, obj: dict, path: Path) -> None:
        # ponytail: meta is plain JSON types (int/float/bool/str) — no custom default needed
        self._atomic_write(path, lambda tmp: json.dump(obj, open(tmp, "w"), indent=2), suffix=".json.tmp")

    def _atomic_write(self, path: Path, writer, *, suffix: str) -> None:
        """Atomic write: tempfile in save_dir → os.replace; unlink tmp on any failure."""
        fd, tmp = tempfile.mkstemp(dir=self.save_dir, suffix=suffix)
        os.close(fd)
        try:
            writer(tmp)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _list_steps(self) -> list:
        steps = []
        for p in self.save_dir.glob("model_step_*.safetensors"):
            try:
                steps.append(int(p.stem.split("_")[-1]))
            except ValueError:
                pass
        return steps

    def _checkpoint_complete(self, step: int) -> bool:
        return all((self.save_dir / n).exists() for n in [
            f"model_step_{step}.safetensors", f"optim_step_{step}.pt", f"meta_step_{step}.json"])
