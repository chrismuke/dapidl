"""Push the p64 QC review sample into Label Studio (DAPIDL org).

Each task = a DAPI crop (embedded as a data: URI, so no external hosting) with the
grader's grade in `data`, plus a *prediction* carrying the nucleus polygon as a
toggleable overlay layer. Henrik flags problems + comments; export joins by `row_idx`.

Usage (run from the p64 worktree, after qc_review_ls_build.py):
    source ~/.zshrc.local
    # create a fresh project + import:
    uv run --with requests python scripts/qc_review_ls_push.py --create
    # update an existing project's config + replace all its tasks:
    uv run --with requests python scripts/qc_review_ls_push.py --project-id 5 --update-config --delete-existing
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import polars as pl
import requests

DSET = Path("/mnt/work/datasets/derived/breast-pilot-6source-dapi-p64-nuc-v1")
OUT = DSET / "qc" / "review_ls"
B = os.environ["LABEL_STUDIO_URL"].rstrip("/")
H = {"Authorization": f"Token {os.environ['LABEL_STUDIO_DAPIDL_API_TOKEN']}"}
DISP = 256
BATCH = 40
PROJECT_TITLE = "p64 QC Scoring Review — DAPI nucleus"
DESCRIPTION = ("Review the automated p64 QC grade. Overlay: cyan = nucleus (toggle with the "
               "eye icon). Leave flags empty if the grade looks right; otherwise flag + comment.")

CONFIG = """<View>
  <View style="display:flex">
    <View style="flex:68%">
      <Image name="img" value="$image" zoom="true" zoomControl="true"
             brightnessControl="true" contrastControl="true"/>
      <PolygonLabels name="segmentation" toName="img" strokeWidth="2"
                     opacity="0.4" fillOpacity="0.1">
        <Label value="nucleus" background="#00e5ff"/>
      </PolygonLabels>
    </View>
    <View style="flex:32%; padding-left:14px">
      <Header value="Grader grade: $assigned_group"/>
      <Header value="$slide  ·  $cell_class  ·  row $row_idx" size="6"/>
      <Header value="anomaly pct: $anomaly_pct" size="6"/>
      <Text name="hint" value="Toggle the nucleus layer with the eye icon. Leave all unchecked if the grade looks right."/>
      <Choices name="flag" toName="img" choice="multiple">
        <Choice value="Wrong grade" hotkey="1"/>
        <Choice value="Bad nucleus segmentation" hotkey="2"/>
        <Choice value="No / wrong nucleus" hotkey="3"/>
        <Choice value="No subnuclear structure visible" hotkey="4"/>
        <Choice value="Other problem" hotkey="5"/>
      </Choices>
      <View visibleWhen="choice-selected" whenTagName="flag" whenChoiceValue="Wrong grade">
        <Text name="cgq" value="Correct grade should be:"/>
        <Choices name="correct_group" toName="img" choice="single-radio">
          <Choice value="Excellent"/><Choice value="Good"/><Choice value="Weak-passing"/>
          <Choice value="Broken-geom"/><Choice value="Broken-quality"/>
        </Choices>
      </View>
      <TextArea name="comment" toName="img" rows="4" editable="true"
                placeholder="Comment on the QC score / why it is wrong (optional)"/>
    </View>
  </View>
</View>"""


def poly_region(rid: int, pts: list, label: str) -> dict:
    return {
        "id": f"{label}_{rid}", "type": "polygonlabels",
        "from_name": "segmentation", "to_name": "img",
        "original_width": DISP, "original_height": DISP, "image_rotation": 0,
        "value": {"points": pts, "polygonlabels": [label], "closed": True},
    }


def build_task(r: dict, sample_dir: Path) -> dict:
    ri = int(r["row_idx"])
    uri = "data:image/png;base64," + base64.b64encode((sample_dir / f"{ri}.png").read_bytes()).decode()
    res = []
    if r["nuc_points"]:
        res.append(poly_region(ri, json.loads(r["nuc_points"]), "nucleus"))
    ap_val = r.get("anomaly_pct")
    data = {"image": uri, "row_idx": ri, "slide": r["slide"],
            "cell_class": r["cell_class"], "assigned_group": r["assigned_group"],
            "anomaly_pct": "" if ap_val is None or ap_val != ap_val else round(float(ap_val), 1)}
    task = {"data": data}
    if res:
        task["predictions"] = [{"model_version": "qc_seg_v3", "result": res}]
    return task


def delete_all_tasks(pid: int) -> None:
    r = requests.post(f"{B}/api/dm/actions", headers=H,
                      params={"id": "delete_tasks", "project": pid},
                      json={"selectedItems": {"all": True, "excluded": []}})
    r.raise_for_status()
    print(f"[push] deleted existing tasks: {r.json()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true", help="create a new project")
    ap.add_argument("--project-id", type=int, help="operate on an existing project")
    ap.add_argument("--update-config", action="store_true", help="PATCH config+description on --project-id")
    ap.add_argument("--delete-existing", action="store_true", help="delete all tasks before import")
    ap.add_argument("--no-import", action="store_true", help="skip importing tasks")
    ap.add_argument("--limit", type=int, default=0, help="only push N tasks (0=all)")
    ap.add_argument("--start", type=int, default=0, help="skip the first N manifest rows")
    ap.add_argument("--sample-dir", type=Path, default=OUT, help="dir with PNGs + manifest.parquet")
    ap.add_argument("--title", type=str, default=PROJECT_TITLE)
    a = ap.parse_args()
    sample_dir = a.sample_dir

    if a.create:
        r = requests.post(f"{B}/api/projects", headers=H, json={
            "title": a.title, "label_config": CONFIG, "description": DESCRIPTION})
        r.raise_for_status()
        pid = r.json()["id"]
        print(f"[push] created project {pid}: '{a.title}'  -> {B}/projects/{pid}/data", flush=True)
    else:
        pid = a.project_id
        if not pid:
            raise SystemExit("need --create or --project-id")

    if a.update_config:
        r = requests.patch(f"{B}/api/projects/{pid}", headers=H,
                           json={"label_config": CONFIG, "description": DESCRIPTION})
        r.raise_for_status()
        print(f"[push] updated config + description on project {pid}", flush=True)

    if a.delete_existing:
        delete_all_tasks(pid)

    if a.no_import:
        print("[push] --no-import: done.", flush=True)
        return

    man = pl.read_parquet(sample_dir / "manifest.parquet")
    rows = man
    if a.start:
        rows = rows.slice(a.start, None)
    if a.limit:
        rows = rows.head(a.limit)

    tasks = [build_task(r, sample_dir) for r in rows.iter_rows(named=True)]
    print(f"[push] importing {len(tasks)} tasks to project {pid} (batch={BATCH})", flush=True)
    done = 0
    for i in range(0, len(tasks), BATCH):
        chunk = tasks[i:i + BATCH]
        rr = requests.post(f"{B}/api/projects/{pid}/import", headers=H, json=chunk)
        rr.raise_for_status()
        done += len(chunk)
        print(f"[push] {done}/{len(tasks)}  resp={rr.json() if i == 0 else 'ok'}", flush=True)
    print(f"[push] DONE — open {B}/projects/{pid}/data", flush=True)


if __name__ == "__main__":
    main()
