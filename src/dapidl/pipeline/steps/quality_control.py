"""Post-hoc quality-control pass over a built DAPIDL dataset.

Reads patches, groups by slide, fits a per-slide normalization reference from a
sample, scores every patch, and writes a sidecar qc/qc_scores.parquet (plus
provenance). metadata.parquet is never modified. Optionally logs montages and
score histograms to ClearML.
"""

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from starpose.qc.classical import ClassicalQualityScorer

from dapidl.qc.io import read_patches
from dapidl.qc.montage import build_class_montage

REFERENCE_SAMPLE = 2000  # patches sampled per slide to fit the normalization ref


def _load_patch_labels(dataset_path: Path):
    """Return (n, cell_ids, class_names) supporting two dataset layouts.

    - PatchExtractor/Zarr: metadata.parquet with cell_id + broad_category/predicted_type
    - LMDB-derived: labels.npy (int) + class_mapping.json (name->int); cell_id = index
    """
    meta_path = dataset_path / "metadata.parquet"
    if meta_path.exists():
        meta = pl.read_parquet(meta_path)
        label_col = "broad_category" if "broad_category" in meta.columns else "predicted_type"
        return meta.height, meta["cell_id"].to_list(), meta[label_col].to_numpy()
    labels_path = dataset_path / "labels.npy"
    mapping_path = dataset_path / "class_mapping.json"
    if labels_path.exists() and mapping_path.exists():
        labels = np.load(labels_path)
        mapping = json.loads(mapping_path.read_text())
        inv = {int(v): k for k, v in mapping.items()}
        class_names = np.array([inv[int(x)] for x in labels], dtype=object)
        return len(labels), list(range(len(labels))), class_names
    raise FileNotFoundError(
        f"{dataset_path} has neither metadata.parquet nor labels.npy+class_mapping.json"
    )


def _slide_groups(dataset_path: Path, n: int) -> np.ndarray:
    """Per-patch slide labels, reconstructed from slide_stats.json (safe JSON).

    Falls back to a single group if slide_stats.json is absent.
    """
    stats_path = dataset_path / "slide_stats.json"
    if not stats_path.exists():
        return np.array(["__single__"] * n, dtype=object)
    stats = json.loads(stats_path.read_text())
    sources = np.empty(n, dtype=object)
    i = 0
    for s in stats:
        cnt = int(s["n_written"])
        sources[i : i + cnt] = s["source"]
        i += cnt
    if i != n:
        raise ValueError(f"slide_stats.json sums to {i} != {n} metadata rows")
    return sources


def run_quality_control(
    dataset_path: Path | str,
    use_clearml: bool = True,
    montage_top_n: int = 64,
    seed: int = 42,
) -> Path:
    """Score a dataset, write the sidecar, build montages. Returns the qc/ dir."""
    dataset_path = Path(dataset_path)
    n, cell_ids, class_names = _load_patch_labels(dataset_path)

    sources = _slide_groups(dataset_path, n)
    scorer = ClassicalQualityScorer()
    rng = np.random.default_rng(seed)
    focus = np.zeros(n)
    detection = np.zeros(n)
    qc = np.zeros(n)
    raw = {k: np.zeros(n) for k in ("var_laplacian", "tenengrad", "foreground_frac", "central_blob_frac")}

    for slide in sorted(set(sources.tolist())):
        idx = np.where(sources == slide)[0]
        sample = idx if len(idx) <= REFERENCE_SAMPLE else rng.choice(idx, REFERENCE_SAMPLE, replace=False)
        ref = scorer.fit_reference(read_patches(dataset_path, sample))
        for start in range(0, len(idx), 1000):
            chunk = idx[start : start + 1000]
            scores = scorer.score_batch(read_patches(dataset_path, chunk), ref=ref)
            for j, gi in enumerate(chunk):
                s = scores[j]
                focus[gi], detection[gi], qc[gi] = s.focus_score, s.detection_score, s.qc_score
                for k in raw:
                    raw[k][gi] = s.metrics[k]
        logger.info(f"QC scored slide {slide}: {len(idx)} patches")

    out_dir = dataset_path / "qc"
    out_dir.mkdir(exist_ok=True)

    scores_df = pl.DataFrame({
        "cell_id": cell_ids,
        "focus_score": focus,
        "detection_score": detection,
        "qc_score": qc,
        "var_laplacian": raw["var_laplacian"],
        "tenengrad": raw["tenengrad"],
        "foreground_frac": raw["foreground_frac"],
        "central_blob_frac": raw["central_blob_frac"],
    })
    _atomic_write_parquet(scores_df, out_dir / "qc_scores.parquet")

    (out_dir / "qc_scores.meta.json").write_text(json.dumps({
        "scorer": scorer.name,
        "params": {"varlap_floor": scorer.varlap_floor, "fg_lo": scorer.fg_lo, "fg_hi": scorer.fg_hi},
        "reference_sample": REFERENCE_SAMPLE,
        "date": date.today().isoformat(),
    }, indent=2))

    montages = _build_montages(dataset_path, class_names, qc, montage_top_n, out_dir)
    if use_clearml:
        _log_to_clearml(dataset_path, scores_df, montages)
    return out_dir


def _atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))


def _build_montages(dataset_path, class_names, qc, top_n, out_dir) -> dict:
    """One worst-first montage PNG per cell type. Returns {cell_type: image}.

    Selects the worst top_n patches by qc score BEFORE reading pixels.
    """
    montages = {}
    import matplotlib.image as mpimg
    for cell_type in sorted(set(class_names.tolist())):
        idx = np.where(class_names == cell_type)[0]
        if len(idx) == 0:
            continue
        worst = idx[np.argsort(qc[idx])[:top_n]]
        patches = read_patches(dataset_path, worst)
        img = build_class_montage(patches, qc[worst], cell_type, top_n=top_n)
        mpimg.imsave(out_dir / f"montage_{_safe_name(cell_type)}.png", img)
        montages[cell_type] = img
    return montages


def _log_to_clearml(dataset_path, scores_df, montages) -> None:
    import clearml

    task = clearml.Task.current_task() or clearml.Task.init(
        project_name="DAPIDL/QC",
        task_name=f"qc_{dataset_path.name}",
        task_type=clearml.Task.TaskTypes.qc,
    )
    qc_logger = task.get_logger()
    for cell_type, img in montages.items():
        qc_logger.report_image(title="worst_qc", series=cell_type, iteration=0, image=img)
    qc_logger.report_histogram(
        title="qc_score", series="all", iteration=0,
        values=scores_df["qc_score"].to_numpy(),
    )
    task.upload_artifact(name="qc_scores", artifact_object=scores_df.to_pandas())
