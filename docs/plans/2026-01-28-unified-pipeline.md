# Unified 1-N Dataset Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge single-dataset (`clearml-pipeline run`) and multi-dataset (`clearml-pipeline universal`) into one pipeline command that supports 1-N datasets via `-t` flags, with Cell Ontology standardization always on.

**Architecture:** The existing `UnifiedPipelineController` already handles both single and multi-tissue modes via `is_multi_tissue`. We unify the CLI to always use `-t` flags and route through this controller. For N=1 with no tissue name, we auto-detect tissue from platform/data. Training always uses `MultiTissueDataset` (works fine with 1 dataset) for consistent Cell Ontology mapping.

**Tech Stack:** Click (CLI), ClearML (pipeline orchestration), PyTorch (training), LMDB (datasets)

---

## Key Design Decisions

1. **Always `-t` flags** - `dapidl pipeline run -t lung bf8f913f xenium 2`
2. **Cell Ontology always on** - even for single datasets
3. **One training path** - `MultiTissueDataset` handles both N=1 and N>1
4. **Keep old commands as deprecated aliases** - don't break existing scripts
5. **`-t` accepts dataset ID OR local path** - auto-detect based on path existence

---

### Task 1: Unify CLI Command

**Files:**
- Modify: `src/dapidl/cli.py` - Replace `run` command, deprecate `universal`

**Step 1: Replace `clearml-pipeline run` with unified command**

The new `run` command uses `-t` for all dataset input:

```python
@clearml_pipeline_group.command(name="run")
@click.option(
    "--tissue", "-t",
    multiple=True,
    nargs=4,
    type=(str, str, str, int),
    help="Add dataset: TISSUE SOURCE PLATFORM TIER. SOURCE is ClearML dataset ID or local path.",
)
@click.option("--sampling", type=click.Choice(["equal", "proportional", "sqrt"]), default="sqrt")
@click.option("--segmenter", type=click.Choice(["cellpose", "native"]), default="native")
@click.option("--annotator", type=click.Choice(["celltypist", "ground_truth", "popv"]), default="celltypist")
@click.option("--patch-size", type=click.Choice(["32", "64", "128", "256"]), default="128")
@click.option("--backbone", default="efficientnetv2_rw_s")
@click.option("--epochs", type=int, default=50)
@click.option("--batch-size", type=int, default=64)
@click.option("--local", is_flag=True)
@click.option("--project", default="DAPIDL/pipelines")
@click.option("--skip-training", is_flag=True)
@click.option("--fine-grained", is_flag=True)
@click.option("--validate", is_flag=True)
def run_pipeline(tissue, sampling, segmenter, annotator, patch_size, backbone,
                 epochs, batch_size, local, project, skip_training, fine_grained, validate):
```

Key logic in the function body:
- Validate at least one `-t` provided
- For each tissue tuple `(name, source, platform, tier)`:
  - If `source` is a path that exists on disk → use as `local_path`
  - Otherwise → use as `dataset_id`
  - Call `config.input.add_tissue(...)`
- Create `UnifiedPipelineController(config)`
- Route to local or remote execution

**Step 2: Add deprecated alias for `universal` command**

Keep `universal` as a thin wrapper that prints deprecation warning and calls `run_pipeline`.

**Step 3: Run lint check**

Run: `uv run ruff check src/dapidl/cli.py`

**Step 4: Commit**

```
feat: unify pipeline CLI to support 1-N datasets via -t flags
```

---

### Task 2: Route Single-Dataset Through UnifiedPipelineController

**Files:**
- Modify: `src/dapidl/pipeline/unified_controller.py`

**Current state:** `UnifiedPipelineController` already has `is_multi_tissue` that checks `len(self.config.input.tissues) > 0`. When tissues list has 1 entry, it takes the multi-tissue path which works but uses the heavier `UniversalDAPITrainingStep`.

**Changes needed:**

**Step 1: Always use multi-tissue path for training**

In `_add_single_dataset_steps()`, the single-dataset steps (data_loader → segmentation → annotation → LMDB) stay the same. But instead of the basic `TrainingStep`, route to `UniversalDAPITrainingStep` which uses `MultiTissueDataset` and Cell Ontology.

The simplest approach: when `len(tissues) == 1`, still run the single-dataset data prep steps (they're simpler/faster for 1 dataset), but use the universal training step.

Modify `_add_single_dataset_steps()` training section (around line 261) to use universal training step with `standardize_labels=True`.

**Step 2: Ensure Cell Ontology is always enabled**

In `DAPIDLPipelineConfig` default or in the CLI, set `ontology.enabled = True` by default. Check `unified_config.py` `OntologyConfig` defaults.

**Step 3: Test locally with single dataset**

```bash
uv run dapidl clearml-pipeline run -t lung bf8f913f000a43d4b4d8bb8f0c0c5284 xenium 2 --local --epochs 5
```

**Step 4: Commit**

```
feat: route single-dataset training through MultiTissueDataset
```

---

### Task 3: Make MultiTissueDataset Work Cleanly With N=1

**Files:**
- Modify: `src/dapidl/data/multi_tissue_dataset.py` (if needed)

**Current state:** `MultiTissueDataset` already works with 1 dataset. But verify:
- `create_multi_tissue_splits()` handles N=1 without errors
- Sampling strategy with 1 tissue doesn't cause division by zero
- Cell Ontology mapping works with single tissue

**Step 1: Read and verify N=1 path in MultiTissueDataset**

Check `_build_unified_labels()`, `get_sample_weights()`, and `create_multi_tissue_splits()` for any N>1 assumptions.

**Step 2: Fix any N=1 edge cases found**

Likely areas:
- Tissue-balanced sampling with 1 tissue (sqrt(1) = 1, should be fine)
- `standardize_labels=True` with single dataset (should just map through CL)

**Step 3: Commit if changes made**

```
fix: ensure MultiTissueDataset handles N=1 cleanly
```

---

### Task 4: Update UniversalDAPITrainingStep for Single-Dataset Inputs

**Files:**
- Modify: `src/dapidl/pipeline/steps/universal_training.py`

**Current state:** `UniversalDAPITrainingStep.execute()` expects `dataset_configs` list or multiple `patches_path_N` artifacts. When coming from single-dataset pipeline, it receives a single `patches_path` or `lmdb_path`.

**Step 1: Handle single-dataset artifacts in universal training step**

In `execute()`, after existing dataset resolution, add fallback:

```python
# If no multi-dataset configs found, treat single dataset as 1-tissue config
if not mt_config.datasets:
    single_path = inputs.get("dataset_path") or inputs.get("lmdb_path") or inputs.get("patches_path")
    if single_path:
        resolved = resolve_artifact_path(single_path, "single_dataset")
        mt_config.add_dataset(
            path=str(resolved),
            tissue=inputs.get("tissue", "unknown"),
            platform=inputs.get("platform", "xenium"),
            confidence_tier=int(inputs.get("confidence_tier", 2)),
        )
```

**Step 2: Pass tissue metadata from pipeline controller to training step**

In `unified_controller.py` `_add_single_dataset_steps()`, when adding the training step, include tissue/platform info in `parameter_override`:

```python
"step_config/tissue": tissue_name,
"step_config/platform": cfg.input.platform.value,
"step_config/confidence_tier": cfg.input.tissues[0].confidence_tier if cfg.input.tissues else 2,
```

**Step 3: Commit**

```
feat: universal training step handles single-dataset input
```

---

### Task 5: Update Base Tasks and Test Remote Execution

**Files:**
- No code changes

**Step 1: Recreate ClearML base tasks**

```bash
uv run dapidl clearml-pipeline create-base-tasks --project "DAPIDL/pipelines"
```

**Step 2: Test single-dataset remote pipeline**

```bash
uv run dapidl clearml-pipeline run -t lung bf8f913f000a43d4b4d8bb8f0c0c5284 xenium 2 --epochs 10
```

**Step 3: Test multi-dataset remote pipeline**

```bash
uv run dapidl clearml-pipeline run \
  -t lung bf8f913f000a43d4b4d8bb8f0c0c5284 xenium 2 \
  -t heart 482be038e6224fa7828128dd106bb42f xenium 2 \
  --epochs 10 --sampling sqrt
```

**Step 4: Monitor both on ClearML and verify completion**

**Step 5: Commit any fixes**

---

### Task 6: Clean Up Deprecated Code

**Files:**
- Modify: `src/dapidl/cli.py` - Mark old commands deprecated
- Modify: `CLAUDE.md` - Update pipeline documentation

**Step 1: Add deprecation warnings to old `universal` command**

```python
@clearml_pipeline_group.command(name="universal", deprecated=True)
```

**Step 2: Update CLAUDE.md Commands section**

Replace separate `pipeline` and `universal` examples with unified `-t` syntax.

**Step 3: Commit**

```
docs: update pipeline docs for unified 1-N dataset command
```

---

## Execution Order

Tasks 1-4 are sequential (each builds on previous). Task 5 requires all code changes. Task 6 is cleanup.

## Verification Checklist

- [ ] `dapidl clearml-pipeline run -t lung ID xenium 2 --local --epochs 5` works (N=1)
- [ ] `dapidl clearml-pipeline run -t lung ID xenium 2 -t heart ID2 xenium 2 --local --epochs 5` works (N=2)
- [ ] Cell Ontology labels applied for N=1
- [ ] Per-dataset normalization works for N=2 with different platforms
- [ ] Remote ClearML execution works for both N=1 and N=2
- [ ] Old `universal` command still works with deprecation warning
