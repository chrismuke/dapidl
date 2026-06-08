"""Build manual-QA spreadsheets (one per patch size) + flat-named hardlink mirrors.

For every existing collage PNG in pipeline_output/pilot_qc_collages_v2/collages
we write:
  - A short flat-named hardlink under collages_flat/p{size}/<short>.png so the
    files can be browsed alphabetically without nested directories.
  - One row in qa_p{size}.xlsx with the QA context (slide, class, qc_group,
    n_total, n_shown, both paths) plus two empty user-input columns for
    `patch_ids_to_check` and `commentary`.

Flat filename: `<slide_alias>_<granularity_alias>_<class>_<group_alias>.png`
e.g. `r1_c_Epithelial_ex.png`, `s6_m_Fibroblast_Myofibroblast_we.png`.

Usage:
    uv run python scripts/make_pilot_qa_excel.py
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import polars as pl
import xlsxwriter
from loguru import logger

DEFAULT_ROOT = Path("/mnt/work/git/dapidl/pipeline_output/pilot_qc_collages_v2")

SLIDE_ALIAS = {
    "xenium_rep1_nuc": "r1", "xenium_rep2_nuc": "r2",
    "sthelar_breast_s0": "s0", "sthelar_breast_s1": "s1",
    "sthelar_breast_s3": "s3", "sthelar_breast_s6": "s6",
}
GRAN_ALIAS = {"coarse": "c", "medium": "m"}
GROUP_ALIAS = {
    "Excellent": "ex", "Good": "go", "Weak-passing": "we",
    "Broken-geom": "bg", "Broken-quality": "bq",
}
GROUP_PREFIX = {  # matches render_collage filenames: 01_Excellent, 02_Good, ...
    "Excellent": "01_Excellent", "Good": "02_Good",
    "Weak-passing": "03_Weak-passing", "Broken-geom": "04_Broken-geom",
    "Broken-quality": "05_Broken-quality",
}


def _safe(name: str) -> str:
    """Sanitize cell-type names for use in filenames: +/space/slash -> _."""
    return re.sub(r"[+/ ]", "_", name)


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _orig_class_dir_name(cls: str) -> str:
    """Mirror the rendering script's `cls.replace('/', '_').replace(' ', '_')`."""
    return cls.replace("/", "_").replace(" ", "_")


def _flat_name(slide: str, gran: str, cls: str, group: str) -> str:
    return f"{SLIDE_ALIAS[slide]}_{GRAN_ALIAS[gran]}_{_safe(cls)}_{GROUP_ALIAS[group]}.png"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()

    counts = pl.read_parquet(args.root / "counts.parquet")
    # Only rows where a collage was actually rendered.
    counts = counts.filter(pl.col("n_total") > 0)
    logger.info(f"counts rows w/ collage: {counts.height}")

    flat_root = args.root / "collages_flat"
    flat_root.mkdir(exist_ok=True)

    for size in (128, 64):
        sheet_rows: list[dict] = []
        size_flat_dir = flat_root / f"p{size}"
        size_flat_dir.mkdir(exist_ok=True)

        sub = counts.filter(pl.col("size") == size).sort(
            ["granularity", "slide", "cell_type", "qc_group"]
        )

        for r in sub.iter_rows(named=True):
            gran = r["granularity"]
            slide = r["slide"]
            cls = r["cell_type"]
            group = r["qc_group"]
            orig = (args.root / "collages" / f"p{size}" / gran / slide
                    / _orig_class_dir_name(cls) / f"{GROUP_PREFIX[group]}.png")
            if not orig.exists():
                # Should be unreachable given the n_total > 0 filter, but defend.
                logger.warning(f"missing PNG: {orig}")
                continue
            flat = size_flat_dir / _flat_name(slide, gran, cls, group)
            _hardlink_or_copy(orig, flat)
            sheet_rows.append({
                "size": f"p{size}",
                "granularity": gran,
                "slide": slide,
                "cell_type": cls,
                "qc_group": group,
                "n_total": int(r["n_total"]),
                "n_shown": int(r["n_shown"]),
                "folder_path": str(orig.relative_to(args.root.parent.parent)),
                "flat_filename": flat.name,
            })

        # Write the workbook for this size.
        xlsx_path = args.root / f"qa_p{size}.xlsx"
        wb = xlsxwriter.Workbook(str(xlsx_path))
        ws = wb.add_worksheet("collages")
        headers = ["id", "size", "granularity", "slide", "cell_type", "qc_group",
                   "n_total", "n_shown", "folder_path", "flat_filename",
                   "patch_ids_to_check", "commentary"]
        hdr_fmt = wb.add_format({"bold": True, "bg_color": "#DDDDDD",
                                 "border": 1, "align": "left"})
        for c, h in enumerate(headers):
            ws.write(0, c, h, hdr_fmt)
        for i, row in enumerate(sheet_rows, start=1):
            ws.write(i, 0, i)
            ws.write(i, 1, row["size"])
            ws.write(i, 2, row["granularity"])
            ws.write(i, 3, row["slide"])
            ws.write(i, 4, row["cell_type"])
            ws.write(i, 5, row["qc_group"])
            ws.write(i, 6, row["n_total"])
            ws.write(i, 7, row["n_shown"])
            ws.write(i, 8, row["folder_path"])
            ws.write(i, 9, row["flat_filename"])
            # leave id-list and commentary blank for the user
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(sheet_rows), len(headers) - 1)
        # Friendly column widths.
        widths = [4, 5, 11, 19, 24, 14, 8, 8, 78, 38, 30, 60]
        for c, wpx in enumerate(widths):
            ws.set_column(c, c, wpx)
        wb.close()
        logger.info(f"wrote {xlsx_path}  ({len(sheet_rows)} rows)")
        logger.info(f"      flat dir {size_flat_dir}  "
                    f"({len(list(size_flat_dir.glob('*.png')))} files)")


if __name__ == "__main__":
    main()
