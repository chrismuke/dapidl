#!/usr/bin/env python3
"""Create a draft ClearML Pipeline for Web UI configuration.

This creates a pipeline in ClearML's Pipelines UI where you can:
1. See the pipeline DAG visually
2. Configure all parameters before running
3. Clone and run with different configurations
"""

from clearml import Task
from loguru import logger


def main():
    logger.info("Creating draft ClearML Pipeline Task for Web UI configuration...")

    # Create a draft task with all parameters exposed
    task = Task.init(
        project_name="DAPIDL/universal",
        task_name="DAPIDL Configurable Pipeline (DRAFT)",
        task_type=Task.TaskTypes.controller,
        reuse_last_task_id=False,
    )

    # Connect all configurable parameters
    params = {
        # Dataset selection
        "datasets": "xenium_breast_rep1,xenium_breast_rep2",

        # Model architecture
        "backbone": "efficientnetv2_rw_s",

        # Training parameters
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.0001,

        # Patch extraction
        "patch_size": 128,

        # Segmentation
        "segmenter": "native",

        # Annotation
        "annotator": "celltypist",
        "celltypist_models": "Cells_Adult_Breast.pkl,Immune_All_High.pkl",

        # Multi-tissue training
        "sampling_strategy": "sqrt",
        "standardize_labels": True,

        # Curriculum learning
        "coarse_only_epochs": 20,
        "coarse_medium_epochs": 50,

        # Execution
        "gpu_queue": "gpu",
        "output_dir": "experiment_custom",
    }

    # Connect parameters to task (visible in Web UI)
    task.connect(params, name="Pipeline Configuration")

    # Add helpful description
    task.set_comment("""
DAPIDL Configurable Pipeline - DRAFT

This is a template task. To run a pipeline:
1. Clone this task (right-click → Clone)
2. Edit the "Pipeline Configuration" parameters
3. Enqueue the cloned task to the 'gpu' queue

Available datasets:
- xenium_breast_rep1, xenium_breast_rep2
- xenium_colon_cancer, xenium_colon_normal
- xenium_colorectal_cancer
- xenium_heart_normal
- xenium_kidney_cancer, xenium_kidney_normal
- xenium_liver_cancer, xenium_liver_normal
- xenium_lung_2fov, xenium_lung_cancer
- xenium_lymph_node
- xenium_ovarian_cancer, xenium_ovary_cancer_ff
- xenium_pancreas_cancer
- xenium_skin_sample1, xenium_skin_sample2
- xenium_tonsil_lymphoid, xenium_tonsil_reactive
- merscope_breast

Available backbones:
- efficientnetv2_rw_s (default, good balance)
- efficientnetv2_rw_m (larger)
- convnext_tiny, convnext_small
- resnet50, resnet101

Available annotators:
- celltypist (default)
- popv (ensemble)
- singler (R-based)

Available segmenters:
- native (use platform's segmentation)
- cellpose (deep learning segmentation)
""")

    # Upload the execution script
    task.upload_artifact(
        name="pipeline_script",
        artifact_object="run_customizable_pipeline.py",
    )

    # Mark task as draft (don't execute)
    task.mark_stopped()

    logger.info("=" * 60)
    logger.info("Draft pipeline task created!")
    logger.info("")
    logger.info(f"Task ID: {task.id}")
    logger.info(f"Task URL: https://app.clear.ml/projects/*/experiments/{task.id}")
    logger.info("")
    logger.info("To run a configured pipeline:")
    logger.info("1. Go to ClearML Web UI")
    logger.info("2. Find 'DAPIDL Configurable Pipeline (DRAFT)'")
    logger.info("3. Click Clone to create a copy")
    logger.info("4. Edit 'Pipeline Configuration' parameters")
    logger.info("5. Click Enqueue → select 'gpu' queue")
    logger.info("=" * 60)

    return task.id


if __name__ == "__main__":
    main()
