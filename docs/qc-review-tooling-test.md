# QC Review Tooling Test — Operator Runbook

**Goal:** Stand up the *same* stratified sample of p64 QC patches in **both Label Studio**
and **FiftyOne**, have a (non-technical) reviewer flag mis-graded patches with a reason,
export structured verdicts keyed by `row_idx`, and compare the two tools for ease + output
quality. Run everything with `uv run`. Do **not** touch the GPU.

Run from the p64 worktree: `/mnt/work/git/dapidl-p64qc` (branch `feat/p64-qc-collages`).

---

## Data you will use (read first)

`DSET = /mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1`

| Path | What it is |
|------|------------|
| `$DSET/patches.lmdb` | 17,990 uint16 64×64 patches. key=`struct.pack(">Q", row_idx)`, value=`8-byte header + raw uint16 bytes` |
| `$DSET/patch_registry.parquet` | `row_idx, slide, cell_id, raw_label, x0, y0, pixel_size, coarse_idx` |
| `$DSET/qc/seg_scores.parquet` | 31 cols incl `broken, broken_reason, grade, quality_min`, 4 axes, `cell_provenance` |
| `$DSET/qc/masks/` | bit-packed nucleus/cell masks (`nuc_chunk_*.npz`, `cell_chunk_*.npz`) |
| `$DSET/class_mapping.json` | `{Endothelial:0, Epithelial:1, Immune:2, Stromal:3}` |

Reusable, already-tested helpers live in `scripts/pilot_qc_collages_v3.py`:
`render_tile`, `build_mask_index`, `load_mask_from_index`, `load_patch`, `assign_group`.

The QC group of each patch is `assign_group(broken, broken_reason, grade)` ∈
`{Excellent, Good, Weak-passing, Broken-geom, Broken-quality}`. Each rendered patch is a
DAPI crop with a **1px cyan** nucleus outline (StarDist) and **1px magenta** cell outline.

**Label Studio** is self-hosted and remote: `$LABEL_STUDIO_URL` (=`https://labelstudio.chrism.io`),
token `$LABEL_STUDIO_API_TOKEN` (both set by `source ~/.zshrc.local`). It is **v1.22**; project
id 3 ("SOMACROSS") already exists — **create a new project, do not modify id 3**.

---

## Step 0 — Build the shared sample (do once; both tools consume it)

Create `scripts/qc_review_export.py`:

```python
"""Render a stratified sample of p64 QC patches to PNG + manifest (shared by both tools)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import lmdb, polars as pl
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pilot_qc_collages_v3 import (  # reuse tested helpers
    assign_group, build_mask_index, load_mask_from_index, load_patch, render_tile,
)

DSET = Path("/mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1")
OUT = DSET / "qc" / "review_sample"
N_PER, PS = 15, 64   # 15 patches per (slide × coarse class × qc_group) bucket

def main() -> None:
    scores = pl.read_parquet(DSET / "qc" / "seg_scores.parquet")
    reg = pl.read_parquet(DSET / "patch_registry.parquet").select(["row_idx", "slide", "raw_label"])
    idx2coarse = {v: k for k, v in json.loads((DSET / "class_mapping.json").read_text()).items()}

    df = scores.join(reg, on=["row_idx", "slide"], how="left")
    df = df.with_columns([
        pl.Series("qc_group", [assign_group(b, r or "", g or "Weak-passing")
                               for b, r, g in zip(df["broken"], df["broken_reason"], df["grade"])]),
        pl.Series("cell_class", [idx2coarse.get(int(i), "Unknown") for i in df["coarse_idx"]]),
    ])
    df = df.sample(fraction=1.0, shuffle=True, seed=0)
    sampled = df.group_by(["slide", "cell_class", "qc_group"]).head(N_PER)
    print(f"[export] {sampled.height} patches")

    OUT.mkdir(parents=True, exist_ok=True)
    nuc_index = build_mask_index(DSET / "qc" / "masks", "nuc")
    cell_index = build_mask_index(DSET / "qc" / "masks", "cell")
    env = lmdb.open(str(DSET / "patches.lmdb"), readonly=True, lock=False, readahead=False)
    rows = []
    for r in sampled.iter_rows(named=True):
        ri = int(r["row_idx"])
        rgb = render_tile(load_patch(env, ri, PS),
                          load_mask_from_index(nuc_index, ri, PS),
                          load_mask_from_index(cell_index, ri, PS))
        Image.fromarray(rgb).save(OUT / f"{ri}.png")
        rows.append({"row_idx": ri, "slide": r["slide"], "cell_class": r["cell_class"],
                     "assigned_group": r["qc_group"],
                     "quality_min": None if r["quality_min"] is None else float(r["quality_min"]),
                     "broken_reason": r["broken_reason"], "png": str(OUT / f"{ri}.png")})
    env.close()
    pl.DataFrame(rows).write_parquet(OUT / "manifest.parquet")
    print(f"[export] wrote {len(rows)} PNGs + manifest -> {OUT}")

if __name__ == "__main__":
    main()
```

Run: `uv run --with pillow python scripts/qc_review_export.py`
→ ~1,000–1,500 PNGs + `manifest.parquet` in `$DSET/qc/review_sample/`.
(You do **not** need to rate all of them — rate ~50–100 in each tool to judge it.)

---

## Step 1 — Label Studio (remote instance)

**Serve the images.** LS is remote and cannot read `/mnt/work`. Pick one and set `IMAGE_BASE`:
- **Simplest (if LS can reach this host):** `cd $DSET/qc/review_sample && uv run python -m http.server 8077` → `IMAGE_BASE=http://<this-host>:8077`. Test reachability from LS's side first.
- **Robust:** upload PNGs to `s3://dapidl/qc-review/p64/`, add an **S3 source storage** to the project (LS auto-presigns `s3://` URIs at view time), set `data.image` to the `s3://…` URI. (Do **not** client-side presign with `--profile dev`: SSO temp creds make the URL expire when the SSO session ends.)

Create `scripts/qc_review_ls_push.py`:

```python
"""Create a Label Studio project + import the sampled patches as rating tasks."""
import os
from pathlib import Path
import polars as pl, requests

LS = os.environ["LABEL_STUDIO_URL"].rstrip("/")
H = {"Authorization": f"Token {os.environ['LABEL_STUDIO_API_TOKEN']}"}
IMAGE_BASE = os.environ["IMAGE_BASE"].rstrip("/")
SAMPLE = Path("/mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1/qc/review_sample")

CONFIG = """<View>
  <Image name="img" value="$image" zoom="true" zoomControl="true"/>
  <Header value="Grader: $assigned_group   ($slide / $cell_class / row $row_idx)"/>
  <Choices name="verdict" toName="img" choice="single-radio" required="true">
    <Choice value="Agree" hotkey="1"/><Choice value="Wrong" hotkey="2"/>
  </Choices>
  <View visibleWhen="choice-selected" whenTagName="verdict" whenChoiceValue="Wrong">
    <Choices name="correct_group" toName="img" choice="single-radio">
      <Choice value="Excellent"/><Choice value="Good"/><Choice value="Weak-passing"/>
      <Choice value="Broken-geom"/><Choice value="Broken-quality"/>
    </Choices>
    <Choices name="reason" toName="img" choice="single-radio">
      <Choice value="no_nucleus"/><Choice value="wrong_nucleus"/><Choice value="multi_nucleus"/>
      <Choice value="out_of_focus"/><Choice value="clipped_ok"/><Choice value="other"/>
    </Choices>
  </View>
  <TextArea name="comment" toName="img" rows="2" placeholder="why the grader was wrong"/>
</View>"""

def main():
    man = pl.read_parquet(SAMPLE / "manifest.parquet")
    tasks = [{"data": {"image": f"{IMAGE_BASE}/{r['row_idx']}.png", "row_idx": r["row_idx"],
                       "slide": r["slide"], "cell_class": r["cell_class"],
                       "assigned_group": r["assigned_group"]}} for r in man.iter_rows(named=True)]
    pid = requests.post(f"{LS}/api/projects", headers=H,
                        json={"title": "p64 QC review (test)", "label_config": CONFIG}).json()["id"]
    requests.post(f"{LS}/api/projects/{pid}/import", headers=H, json=tasks).raise_for_status()
    print(f"project {pid}: {len(tasks)} tasks -> {LS}/projects/{pid}/data")

if __name__ == "__main__":
    main()
```

Run: `source ~/.zshrc.local && IMAGE_BASE=http://<host>:8077 uv run --with requests python scripts/qc_review_ls_push.py`

**Review:** open `…/projects/<id>/data`, switch to **Grid view**, add **Filter**s on
`data.slide` / `data.cell_class` / `data.assigned_group`, save each as a **Tab** (one bucket
per tab). Click a thumbnail → press `1` (Agree) or `2` (Wrong → pick correct group + reason) → submit.

**Export:** `GET {LS}/api/projects/<id>/export?exportType=JSON_MIN` (or UI → Export → JSON-MIN)
→ `ls_export.json`. Each record is flat: `{image, row_idx, slide, cell_class, assigned_group,
verdict, correct_group, reason, comment}`.

---

## Step 2 — FiftyOne (local)

Create `scripts/qc_review_fiftyone.py`:

```python
import polars as pl, fiftyone as fo
from pathlib import Path
SAMPLE = Path("/mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1/qc/review_sample")
man = pl.read_parquet(SAMPLE / "manifest.parquet")
ds = fo.Dataset("p64_qc_review", overwrite=True)
samples = []
for r in man.iter_rows(named=True):
    s = fo.Sample(filepath=r["png"])
    s["row_idx"] = int(r["row_idx"]); s["slide"] = r["slide"]
    s["cell_class"] = r["cell_class"]; s["assigned_group"] = r["assigned_group"]
    s["correct_group"] = None; s["reason"] = None; s["comment"] = None
    samples.append(s)
ds.add_samples(samples); ds.persistent = True
print("loaded", len(ds))
session = fo.launch_app(ds, remote=True)   # headless host -> SSH tunnel -L 5151:localhost:5151
session.wait()
```

Run: `uv run --with fiftyone --with polars --with pyarrow python scripts/qc_review_fiftyone.py`
(FiftyOne bundles a local MongoDB; first launch is slow. If headless, SSH-tunnel port 5151.)

**Review:** the App opens to a thumbnail grid. Filter by `assigned_group`/`slide`/`cell_class`
in the left sidebar. Flag wrong crops fast: multi-select → tag **`WRONG`**. For the ones you
want to correct, open the sample's **Annotate** tab (OSS ≥1.16) and set `correct_group`,
`reason`, `comment`.

**Export:**
```python
import polars as pl, fiftyone as fo
ds = fo.load_dataset("p64_qc_review")
ids, tags, cg, rs, cm = ds.values(["row_idx", "tags", "correct_group", "reason", "comment"])
pl.DataFrame({"row_idx": ids,
             "flagged_wrong": [("WRONG" in (t or [])) for t in tags],
             "correct_group": cg, "reason": rs, "comment": cm}).write_parquet("fo_export.parquet")
```

---

## Step 3 — Analysis (same join for both)

```python
import json, polars as pl
DSET = "/mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1"

# Label Studio:
v = pl.DataFrame(json.load(open("ls_export.json")))
v = v.with_columns(human=pl.when(pl.col("verdict") == "Agree")
                   .then(pl.col("assigned_group")).otherwise(pl.col("correct_group")))
# FiftyOne: load fo_export.parquet, join manifest for assigned_group, human = assigned_group
#           if not flagged_wrong else correct_group.

man = pl.read_parquet(f"{DSET}/qc/review_sample/manifest.parquet")
j = man.join(v.select(["row_idx", "verdict", "human", "reason", "comment"]), on="row_idx", how="inner")
print("agreement:", (j["verdict"] == "Agree").mean())
print(j.group_by(["assigned_group", "human"]).len().sort("len", descending=True))   # confusion
print(j.filter(pl.col("verdict") == "Wrong").group_by("reason").len().sort("len", descending=True))
```

This is the payload an AI agent reads: grader-vs-human confusion matrix + reason histogram,
joined cleanly on `row_idx`.

---

## Step 4 — Compare the two tools (the actual deliverable)

Fill this in after a short rating session in each:

| | Label Studio | FiftyOne |
|---|---|---|
| Setup time (incl. image serving) | | |
| Could a non-technical coworker do it solo? (1–5) | | |
| Rate speed (crops/min) | | |
| Output cleanliness (single clean join? Y/N) | | |
| Main friction | | |

Expectation to confirm or refute: LS is the smoother *rating queue* (pre-built form, hotkeys,
saved bucket-tabs, one flat record per crop); FiftyOne is the smoother *grid triage / curation*
(bulk-tag a selection, similarity, filtering) but free-text reason is a side field and it is a
new local service.

---

## CVAT — deliberately excluded (here is why)

CVAT (also self-hosted, `https://cvat.chrism.io`) is a **region / polygon / mask + video-tracking
annotator** built for professional drawing work. For *rating pre-binned crops* it is the wrong
shape: classification-per-image needs an attribute/tag mode switch, the UI is dense for a
non-technical reviewer, and exports key on **filename** (CVAT-XML / Datumaro) rather than a clean
stable `row_idx`, so the join back is fuzzier. Keep CVAT in reserve for the **opposite** task —
when you need a human to *draw or fix* nucleus / cell boundaries (segmentation correction). There
it beats both Label Studio and FiftyOne. It is not the tool for this verdict / confusion-matrix
workflow.
```
