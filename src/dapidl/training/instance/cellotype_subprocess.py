"""dapidl-side runner for CelloType inference via subprocess.

CelloType lives in `/home/chrism/cellotype-env` (py3.10 + torch 2.4.1 + detectron2).
dapidl runs in py3.12 + torch 2.5+. They cannot import each other, so we
bridge via npz files on disk and the launcher
`/home/chrism/cellotype-env/bin/run_cellotype.sh`.

Use this for **inference only** (e.g. Phase D test-slide eval, Phase D Xenium
cross-source). Training-via-subprocess would pay ~5 s of process+import
startup per call, which is unworkable for a per-batch loop. For training,
write a `train_*.py` script that runs entirely inside cellotype-env and
reads dataset files (parquet/zarr) directly.

Example:
    runner = CellotypeSubprocessRunner(
        weights=Path("/home/chrism/git/CelloType/models/tissuenet_model_0019999.pth"),
        config=Path("/home/chrism/git/CelloType/configs/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml"),
    )
    preds = runner.predict_tiles(images=[tile_a, tile_b])
    # preds[0]['instance_map'] is (H, W) uint16
"""

import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from loguru import logger

DEFAULT_LAUNCHER = Path("/home/chrism/cellotype-env/bin/run_cellotype.sh")
DEFAULT_WORKER = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "instance_seg"
    / "cellotype_worker.py"
)


class CellotypeSubprocessRunner:
    """Run CelloType inference in a subprocess via the cellotype-env launcher.

    Args:
        weights: path to pretrained `.pth`.
        config: path to MaskDINO YAML config.
        launcher: shell launcher that activates cellotype-env. Defaults to
            `/home/chrism/cellotype-env/bin/run_cellotype.sh`.
        worker: path to `cellotype_worker.py`. Defaults to the one in
            this repo at `scripts/instance_seg/cellotype_worker.py`.
        device: "cuda" or "cpu".
    """

    def __init__(
        self,
        weights: Path,
        config: Path,
        launcher: Path = DEFAULT_LAUNCHER,
        worker: Path = DEFAULT_WORKER,
        device: str = "cuda",
    ) -> None:
        self.weights = Path(weights)
        self.config = Path(config)
        self.launcher = Path(launcher)
        self.worker = Path(worker)
        self.device = device

        for p in (self.weights, self.config, self.launcher, self.worker):
            if not p.exists():
                raise FileNotFoundError(f"missing path: {p}")

    def predict_tiles(
        self,
        images: list[np.ndarray] | Iterable[np.ndarray],
        keep_tmp: bool = False,
    ) -> list[dict]:
        """Run inference on a batch of tile images.

        One subprocess call covers the whole batch (model is loaded once),
        so per-tile cost is amortized over batch size. Use a sensible batch
        (~50–500 tiles) to amortize the ~5–10 s import cost.

        Args:
            images: list/iterable of numpy arrays. Each can be `(H, W) uint16`
                DAPI (will be percentile-normalized → uint8 RGB) OR
                `(H, W, 3) uint8` already-RGB.
            keep_tmp: if True, leaves the tmpdir on disk for inspection
                (the worker stderr already streams to logger).

        Returns:
            list[dict] — one dict per input tile with keys:
                `instance_map: (H, W) uint16`
                `class_per_instance_id: (n,) int32`
                `score_per_instance_id: (n,) float32`
                `n_instances: int`
            Tiles that errored have `instance_map=None` and an `error: str` key.
        """
        images_list = list(images)
        if not images_list:
            return []

        tmpdir = Path(tempfile.mkdtemp(prefix="cellotype_subproc_"))
        in_dir = tmpdir / "in"
        out_dir = tmpdir / "out"
        in_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)
        try:
            for i, img in enumerate(images_list):
                np.savez_compressed(in_dir / f"tile_{i:06d}.npz", image=img)

            cmd = [
                "bash",
                str(self.launcher),
                str(self.worker),
                "--weights",
                str(self.weights),
                "--config",
                str(self.config),
                "--in-dir",
                str(in_dir),
                "--out-dir",
                str(out_dir),
                "--device",
                self.device,
            ]
            logger.debug(f"cellotype subproc cmd: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
            if result.returncode not in (0, 1):
                # 0 = all ok, 1 = some errored (still return per-tile results)
                logger.error(f"cellotype worker failed: {result.stderr}")
                raise RuntimeError(
                    f"cellotype worker exit {result.returncode}: {result.stderr}"
                )
            for line in result.stderr.splitlines():
                logger.debug(f"worker: {line}")

            out: list[dict] = []
            for i in range(len(images_list)):
                key = f"{i:06d}"
                pred_path = out_dir / f"pred_{key}.npz"
                err_path = out_dir / f"error_{key}.txt"
                if pred_path.exists():
                    data = np.load(pred_path)
                    out.append(
                        {
                            "instance_map": data["instance_map"],
                            "class_per_instance_id": data["class_per_instance_id"],
                            "score_per_instance_id": data["score_per_instance_id"],
                            "n_instances": int(data["n_instances"]),
                        }
                    )
                elif err_path.exists():
                    out.append(
                        {
                            "instance_map": None,
                            "error": err_path.read_text(),
                        }
                    )
                else:
                    out.append(
                        {
                            "instance_map": None,
                            "error": f"missing pred_{key}.npz and error_{key}.txt",
                        }
                    )
            return out
        finally:
            if keep_tmp:
                logger.info(f"kept tmpdir: {tmpdir}")
            else:
                shutil.rmtree(tmpdir, ignore_errors=True)
