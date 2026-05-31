#!/usr/bin/env bash
# One well-scheduled NuSPIRe fine-tune — the fair rematch vs EfficientNet (test F1 0.619).
#
# Why this differs from the h2h nuspire@1e-4 arm (which hit 0.572, still rising, ep3):
#   * lr-schedule warmup-cosine  — linear warmup -> SINGLE monotonic cosine to ~0.
#     The h2h used CosineAnnealingWarmRestarts (T_0=5), whose epoch-5 LR re-spike
#     drove the val-F1 oscillation. No restarts here.
#   * freeze-backbone-epochs 1   — head settles for 1 epoch before the ViT encoder
#     joins, so the random head doesn't shock the pretrained weights.
#   * real budget — 16 epochs x 48k samples/ep = 768k samples seen (~1.4x the
#     EfficientNet baseline's 560k), early-stop disabled (patience=epochs) so the
#     cosine fully anneals and we keep the best-val checkpoint.
# Everything else identical to the h2h for comparability (sources, seed, B=64,
# fixed class weights, backbone_norm auto-feeds NuSPIRe its 0.219/0.181 stats).
set -u
cd /mnt/work/git/dapidl
OUT=pipeline_output/nuspire_scheduled_2026_05_31
mkdir -p "$OUT"

echo "===== $(date '+%F %T') :: TRAIN nuspire (warmup-cosine, frozen-warmup) ====="
HF_HUB_DISABLE_PROGRESS_BARS=1 uv run python scripts/breast_pooled_train.py \
  --train-sources xenium_rep1,sthelar_breast_s0,sthelar_breast_s1,sthelar_breast_s3,sthelar_breast_s6 \
  --test-sources xenium_rep2 \
  --tier coarse --batch-size 64 --samples-per-epoch 48000 \
  --epochs 16 --patience 16 --seed 42 --fixed-class-weights --num-workers 12 \
  --backbone nuspire --lr 1e-4 --lr-schedule warmup-cosine \
  --warmup-frac 0.1 --min-lr 1e-6 --freeze-backbone-epochs 1 \
  --output "$OUT/nuspire_warmcos" > "$OUT/nuspire_warmcos.log" 2>&1 \
  && echo "----- $(date '+%F %T') :: DONE" \
  || { echo "!!!!! $(date '+%F %T') :: FAILED (see $OUT/nuspire_warmcos.log)"; exit 1; }

echo "===== $(date '+%F %T') :: SCHEDULED RUN COMPLETE -- vs h2h baselines ====="
uv run python - <<'PY'
import json, glob, os
import polars as pl
import numpy as np
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

rows = []
# the well-scheduled run + the h2h arms, all on held-out xenium_rep2
paths = {
  "efficientnet@3e-4 (baseline)": "pipeline_output/h2h_2026_05_30/efficientnetv2_rw_s",
  "nuspire@1e-4 (h2h)":           "pipeline_output/h2h_2026_05_30/nuspire_lr1e-4",
  "nuspire warmup-cosine (NEW)":  "pipeline_output/nuspire_scheduled_2026_05_31/nuspire_warmcos",
}
print(f"{'arm':32s} {'ep':>3s} {'valF1':>6s} {'testF1':>6s} {'MCC':>6s} {'balAcc':>6s} {'acc':>6s}")
for name, d in paths.items():
    sp = f"{d}/summary.json"
    if not os.path.exists(sp):
        print(f"{name:32s}  (missing {sp})"); continue
    s = json.load(open(sp)); m = s.get("per_test", {}).get("xenium_rep2", {})
    g = lambda x: f"{x:.3f}" if isinstance(x,(int,float)) else "  -  "
    print(f"{name:32s} {s.get('best_epoch','-'):>3} {g(s.get('best_val_macro_f1')):>6s} "
          f"{g(m.get('macro_f1')):>6s} {g(m.get('mcc')):>6s} {g(m.get('balanced_accuracy')):>6s} {g(m.get('accuracy')):>6s}")

CLS = ["Endo","Epi","Imm","Str"]
print("\nper-class F1 on xenium_rep2:")
print(f"{'arm':32s} " + " ".join(f"{c:>6s}" for c in CLS))
for name, d in paths.items():
    pp = f"{d}/preds.parquet"
    if not os.path.exists(pp): continue
    df = pl.read_parquet(pp); yt = df['y_true'].to_numpy(); yp = df['y_pred'].to_numpy()
    perc = f1_score(yt, yp, labels=[0,1,2,3], average=None, zero_division=0)
    print(f"{name:32s} " + " ".join(f"{v:6.3f}" for v in perc))

# paired McNemar: EfficientNet baseline vs the new scheduled NuSPIRe
eff = "pipeline_output/h2h_2026_05_30/efficientnetv2_rw_s/preds.parquet"
new = "pipeline_output/nuspire_scheduled_2026_05_31/nuspire_warmcos/preds.parquet"
if os.path.exists(eff) and os.path.exists(new):
    a = pl.read_parquet(eff).select(["cell_id","y_true","y_pred"]).rename({"y_pred":"yp_eff"})
    b = pl.read_parquet(new).select(["cell_id","y_pred"]).rename({"y_pred":"yp_new"})
    j = a.join(b, on="cell_id", how="inner")
    yt = j['y_true'].to_numpy(); ce = (j['yp_eff'].to_numpy()==yt); cn = (j['yp_new'].to_numpy()==yt)
    b_, c_ = int((ce&~cn).sum()), int((~ce&cn).sum())
    res = mcnemar([[int((ce&cn).sum()), b_],[c_, int((~ce&~cn).sum())]], exact=False, correction=True)
    print(f"\nMcNemar  EfficientNet vs NuSPIRe-scheduled  (n={j.height:,})")
    print(f"  eff-only correct={b_:,}  nuspire-only correct={c_:,}  chi2={res.statistic:.2f}  p={res.pvalue:.3e}")
    print(f"  acc: eff={ce.mean():.4f}  nuspire={cn.mean():.4f}  delta={cn.mean()-ce.mean():+.4f} (NuSPIRe - eff)")
PY
echo "===== done ====="
