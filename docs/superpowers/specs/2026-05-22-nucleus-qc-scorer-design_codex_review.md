# Codex review — Segmentation-Grounded Nucleus QC Scorer spec

- **Date:** 2026-05-22
- **Reviewer:** codex (gpt-5.5, xhigh), spec fed inline
- **Verdict:** **GO-WITH-CHANGES**
- Companion to `2026-05-22-nucleus-qc-scorer-design.md`.

---

The segmentation-grounded direction is technically sound for the failure modes tied to localization: no target nucleus, off-center target, edge-cut nuclei, obvious false detections. It is a major improvement over whole-patch focus metrics because it scores the object that supposedly owns the label.

The weak part is `structure_score` as a proxy for "broken." Flat interior can mean tangential/blurred, but it can also be valid biology, source-specific PSF, staining, saturation, small dense nuclei, apoptotic/pyknotic cells, lymphocyte-like cells, or low-texture cell types. StarDist probability and morphology also encode source/model bias, not just objectness.

Missing edge cases:
- crowded/touching nuclei where the center is inside a merged object;
- valid small or elongated nuclei failing area/eccentricity thresholds;
- slides/FOVs where all patches are bad, making per-slide p90 normalization meaningless;
- platform-specific blur/noise that lowers "structure" for an entire source;
- transcriptomic label mismatch or doublets that are not visible from DAPI alone;
- bad pixel size metadata, saturation, uneven illumination, or severe background;
- original source segmentation disagreeing with StarDist re-segmentation.

There is some over-engineering. Cellpose consensus, `FocusExpert`, entropy plus LoG plus std, composite scoring, ClearML logging, and full pipeline integration are all plausible, but too much for v1. A sharper v1 would be: StarDist center-object selection, geometry/completeness checks, one conservative interior texture metric, montages, and source/class/area audit. Add Cellpose/focus experts only after proving the simpler rejector misses important broken cases.

Heuristic per-axis thresholds without labels can work only as a high-specificity rejector for obvious garbage. They cannot reliably define the boundary between "broken" and "valid but low-quality/biologically different." Visual ladders are necessary but insufficient because they do not quantify false positives by source/class/size. A small labeled calibration set is the practical missing piece, even if only 500-2000 stratified examples with "definitely broken / definitely usable / ambiguous."

The biggest risk is systematic censoring: the scorer may drop real nuclei disproportionately by source, class, morphology, or size, then improve apparent patch cleanliness while biasing the downstream cell-type classifier and worsening cross-platform transfer. The guardrail should be conservative thresholds plus stratified false-positive review, not just broken-rate reporting.

Verdict: GO-WITH-CHANGES.

---

## Controller's take on which changes to adopt

**Adopt (clear wins):**
1. **Simplify v1** — StarDist-only center-object selection + geometry/completeness + **one** conservative interior-texture metric + montages + audit. Defer Cellpose consensus, the `FocusExpert` plug-in, and the multi-metric texture stack until the simple rejector is shown to miss cases. (Removes YAGNI from §4.)
2. **Reframe as a high-specificity rejector of *obvious* broken patches**, not a fine quality boundary — matches the "remove broken only" use. Conservative thresholds; when in doubt, keep.
3. **`structure_score` is the danger axis** — low structure correlates with real biology (small dense lymphocytes, pyknotic/apoptotic nuclei). Make it the *most* conservative axis (extreme-flat only), and never let it be the sole reason to drop a patch without the per-class guardrail.
4. **Guardrail = stratified false-positive review**, not just broken-rate: report broken-rate **by source × class × size bin**, and require that flagged patches are not concentrated in a real cell type (e.g., Immune). Bake into the readout.

**Needs user decision:**
5. **Small labeled calibration set** (codex's "practical missing piece"): ~500–2000 stratified patches tagged definitely-broken / definitely-usable / ambiguous, to set thresholds and *quantify* false positives by source/class/size. The user earlier deprioritized labeling, but this is a smaller, validation-only ask than training a model — and codex argues thresholds-without-labels can only catch obvious garbage. Open question for the user.
