"""Base trainer for instance-segmentation models in dapidl.

Handles the cross-cutting concerns shared by CelloType, starpose+class-head,
and any future joint architecture:

- Per-flight GPU memory check (require ≥4 GB headroom over `est_vram_gb`).
- File-lock at `/tmp/dapidl_seg_train.lock` to prevent concurrent training.
- OOM auto-fallback: catch `torch.cuda.OutOfMemoryError`, halve batch, retry.
  Max 2 retries before abort.
- Top-3 checkpoint retention by val PQ.
- W&B logging (project `dapidl-instance-seg`, group `pilot` or `loto`).
- Resume from checkpoint on restart.
- Early stopping on val PQ patience.

Subclasses implement `train_step`, `eval_step`, `compute_loss`, and
`build_model`.
"""

import contextlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

_TRAIN_LOCK = Path("/tmp/dapidl_seg_train.lock")


@dataclass
class TrainerConfig:
    out_dir: Path
    epochs: int = 50
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-5
    early_stop_patience: int = 10
    est_vram_gb: float = 14.0
    headroom_gb: float = 4.0
    use_wandb: bool = True
    wandb_project: str = "dapidl-instance-seg"
    wandb_group: str = "pilot"
    run_name: str = "run"
    keep_top_k: int = 3
    seed: int = 42
    resume: bool = True
    extra: dict = field(default_factory=dict)


class TrainLockBusyError(RuntimeError):
    """Raised when another instance-seg training is already running."""


@contextlib.contextmanager
def acquire_train_lock():
    """File-based mutex preventing two seg trainings on the same GPU."""
    if _TRAIN_LOCK.exists():
        try:
            existing_pid = int(_TRAIN_LOCK.read_text().strip())
            if _pid_alive(existing_pid):
                raise TrainLockBusyError(
                    f"Another seg training is running (pid={existing_pid}). "
                    f"Wait or remove {_TRAIN_LOCK} if stale."
                )
        except (ValueError, FileNotFoundError):
            pass
    _TRAIN_LOCK.write_text(str(os.getpid()))
    try:
        yield
    finally:
        try:
            _TRAIN_LOCK.unlink()
        except FileNotFoundError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def gpu_preflight(est_vram_gb: float, headroom_gb: float = 4.0) -> None:
    """Abort if GPU free memory < est_vram_gb + headroom_gb."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available; skipping GPU pre-flight check")
        return
    free, _total = torch.cuda.mem_get_info()
    free_gb = free / 1024**3
    required_gb = est_vram_gb + headroom_gb
    if free_gb < required_gb:
        raise RuntimeError(
            f"GPU pre-flight failed: free={free_gb:.1f} GB, "
            f"need {required_gb:.1f} GB (estimate {est_vram_gb} + "
            f"headroom {headroom_gb}). Free up GPU memory or wait."
        )
    logger.info(f"GPU pre-flight ok: free={free_gb:.1f} GB ≥ {required_gb:.1f} GB")


class InstanceSegTrainerBase(ABC):
    """Base class for instance-seg trainers."""

    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self.cfg.out_dir = Path(self.cfg.out_dir)
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self._best_val_pq = -1.0
        self._epochs_no_improve = 0
        self._start_epoch = 0
        self._wandb_run = None
        torch.manual_seed(cfg.seed)

        gpu_preflight(cfg.est_vram_gb, cfg.headroom_gb)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        if cfg.resume:
            self._maybe_resume()

        if cfg.use_wandb:
            self._init_wandb()

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        """Construct the model. Called once in __init__."""
        ...

    @abstractmethod
    def train_step(self, batch: dict) -> dict:
        """One forward+backward step. Returns dict of scalar metrics."""
        ...

    @abstractmethod
    def eval_step(self, batch: dict) -> dict:
        """One eval forward pass. Returns dict including 'pq' (panoptic)."""
        ...

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Main training loop with OOM auto-fallback and lock-protected GPU."""
        with acquire_train_lock():
            for epoch in range(self._start_epoch, self.cfg.epochs):
                t0 = time.time()
                train_metrics = self._run_epoch(train_loader, epoch, train=True)
                val_metrics = self._run_epoch(val_loader, epoch, train=False)
                val_pq = float(val_metrics.get("pq", 0.0))
                logger.info(
                    f"epoch {epoch}: train_loss={train_metrics.get('loss', 0):.4f}, "
                    f"val_pq={val_pq:.4f}, time={time.time()-t0:.1f}s"
                )

                self._save_checkpoint(epoch, val_pq)
                if val_pq > self._best_val_pq:
                    self._best_val_pq = val_pq
                    self._epochs_no_improve = 0
                else:
                    self._epochs_no_improve += 1
                    if self._epochs_no_improve >= self.cfg.early_stop_patience:
                        logger.info(
                            f"early stop: no improvement for {self.cfg.early_stop_patience} epochs"
                        )
                        break

                if self._wandb_run is not None:
                    log_payload = {f"train/{k}": v for k, v in train_metrics.items()}
                    log_payload.update({f"val/{k}": v for k, v in val_metrics.items()})
                    log_payload["epoch"] = epoch
                    self._wandb_run.log(log_payload)

        if self._wandb_run is not None:
            self._wandb_run.finish()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run_epoch(
        self, loader: DataLoader, epoch: int, train: bool
    ) -> dict:
        self.model.train(train)
        agg: dict[str, float] = {}
        n = 0
        for batch in loader:
            with torch.set_grad_enabled(train):
                metrics = self._step_with_oom_fallback(batch, train)
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            n += 1
        return {k: v / max(n, 1) for k, v in agg.items()}

    def _step_with_oom_fallback(
        self, batch: dict, train: bool, max_retries: int = 2
    ) -> dict:
        for attempt in range(max_retries + 1):
            try:
                return self.train_step(batch) if train else self.eval_step(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt == max_retries:
                    raise
                logger.warning(
                    f"OOM, halving batch and retrying (attempt {attempt + 1}/{max_retries})"
                )
                batch = self._halve_batch(batch)
        raise RuntimeError("unreachable")

    @staticmethod
    def _halve_batch(batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] > 1:
                out[k] = v[: v.shape[0] // 2]
            elif isinstance(v, list) and len(v) > 1:
                out[k] = v[: len(v) // 2]
            else:
                out[k] = v
        return out

    def _save_checkpoint(self, epoch: int, val_pq: float) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "val_pq": val_pq,
            "best_val_pq": self._best_val_pq,
            "epochs_no_improve": self._epochs_no_improve,
        }
        path = self.cfg.out_dir / f"ckpt_epoch{epoch:03d}_pq{val_pq:.4f}.pt"
        torch.save(ckpt, path)
        # Prune to top-K by val_pq
        ckpts = sorted(self.cfg.out_dir.glob("ckpt_epoch*.pt"))
        if len(ckpts) > self.cfg.keep_top_k:
            scored = [(self._parse_pq(p), p) for p in ckpts]
            scored.sort(reverse=True)
            for _, p in scored[self.cfg.keep_top_k :]:
                p.unlink(missing_ok=True)

    @staticmethod
    def _parse_pq(path: Path) -> float:
        try:
            return float(path.stem.split("_pq")[-1])
        except (IndexError, ValueError):
            return -1.0

    def _maybe_resume(self) -> None:
        ckpts = sorted(self.cfg.out_dir.glob("ckpt_epoch*.pt"))
        if not ckpts:
            return
        # Resume from highest-epoch checkpoint
        latest = max(ckpts, key=lambda p: int(p.stem.split("_")[1].replace("epoch", "")))
        logger.info(f"resuming from {latest}")
        ckpt = torch.load(latest, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self._start_epoch = int(ckpt["epoch"]) + 1
        self._best_val_pq = float(ckpt["best_val_pq"])
        self._epochs_no_improve = int(ckpt["epochs_no_improve"])

    def _init_wandb(self) -> None:
        try:
            import wandb

            self._wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                group=self.cfg.wandb_group,
                name=self.cfg.run_name,
                config={
                    "epochs": self.cfg.epochs,
                    "batch_size": self.cfg.batch_size,
                    "lr": self.cfg.lr,
                    "weight_decay": self.cfg.weight_decay,
                    "early_stop_patience": self.cfg.early_stop_patience,
                    **self.cfg.extra,
                },
                dir=str(self.cfg.out_dir),
            )
        except Exception as e:
            logger.warning(f"wandb init failed ({e}); continuing without wandb")
            self._wandb_run = None

    def write_summary(self, summary: dict) -> None:
        (self.cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
