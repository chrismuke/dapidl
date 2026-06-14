#!/usr/bin/env python3
"""Phase-1 gate: run derive_labels over the FULL real per-source raw vocabularies
(Janesick-17 for Xenium, ct_tangram for STHELAR) and report coarse/medium results +
any Unknowns. Reviewed before the expensive p64 build."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import polars as pl

sys.path.insert(0, "scripts")
import breast_dapi_lmdb as B  # noqa: E402  inline dicts + STHELAR_BASE
from dapidl.data.sthelar import SthelarDataReader  # noqa: E402
from dapidl.ontology.training_tiers import derive_labels  # noqa: E402

OUT = Path("pipeline_output/pilot_qc_collages_v3")
OUT.mkdir(parents=True, exist_ok=True)


def _sthelar_vocab(zarr) -> Counter:
    ndf = SthelarDataReader(zarr).nucleus_df
    col = "ct_tangram" if "ct_tangram" in ndf.columns else "label1"
    return Counter(str(x) for x in ndf[col].to_list())


def main() -> None:
    rows = []
    for src in ["xenium_rep1", "xenium_rep2"]:
        for raw in sorted(B.JANESICK17_TO_COARSE):
            c, m = derive_labels(raw, src)
            rows.append((src, raw, 0, c, m))
    for z in sorted(B.STHELAR_BASE.glob("sdata_breast_s*.zarr")):
        src = "sthelar_" + z.name.replace("sdata_", "").replace(".zarr", "")
        for raw, n in sorted(_sthelar_vocab(z).items(), key=lambda kv: -kv[1]):
            c, m = derive_labels(raw, src)
            rows.append((src, raw, n, c, m))
    df = pl.DataFrame(rows, schema=["source", "raw", "n_cells", "coarse", "medium"], orient="row")
    df.write_parquet(OUT / "label_divergence.parquet")

    unknown = df.filter(pl.col("coarse") == "Unknown")
    lines = ["# p64 label divergence — derive_labels over full per-source vocabularies\n",
             f"- total (source,raw) pairs: {df.height}",
             f"- coarse==Unknown pairs: {unknown.height}",
             f"- STHELAR cells under Unknown raws: {int(unknown['n_cells'].sum())}\n",
             "## Unknown raws (need a GT mapping if cell counts are non-trivial)\n"]
    if unknown.height:
        for s, r, n, _, _ in unknown.sort("n_cells", descending=True).iter_rows():
            lines.append(f"- `{s}` `{r}` (n={n})")
    else:
        lines.append("- (none — every raw label maps to a coarse class)")
    lines.append("\n## Coarse distribution of mapped (source,raw) pairs\n")
    for c, n in df.filter(pl.col("coarse") != "Unknown").group_by("coarse").len().sort("len", descending=True).iter_rows():
        lines.append(f"- {c}: {n} raw types")
    (OUT / "label_divergence.md").write_text("\n".join(lines))
    print(f"divergence: {df.height} pairs, {unknown.height} Unknown raws "
          f"({int(unknown['n_cells'].sum())} STHELAR cells) -> {OUT/'label_divergence.md'}")


if __name__ == "__main__":
    main()
