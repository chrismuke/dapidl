#!/usr/bin/env python3
"""Create ClearML base tasks for the Universal DAPIDL Pipeline.

These base tasks are templates that the ClearML Pipeline uses to spawn
actual step executions. They must be created before running the pipeline
as a ClearML Pipeline (as opposed to run_locally()).
"""

from loguru import logger

from dapidl.pipeline.universal_controller import (
    UniversalDAPIPipelineController,
    UniversalPipelineConfig,
)


def main():
    logger.info("Creating ClearML base tasks for Universal Pipeline")
    logger.info("=" * 60)

    # Create a minimal config just for base task creation
    config = UniversalPipelineConfig(
        name="dapidl-universal-base-tasks",
        project="DAPIDL/universal",
    )

    # Create controller and generate base tasks
    controller = UniversalDAPIPipelineController(config)
    controller.create_base_tasks()

    logger.info("=" * 60)
    logger.info("Base tasks created successfully!")
    logger.info("")
    logger.info("The following base tasks are now available in ClearML:")
    logger.info("  - step-data_loader")
    logger.info("  - step-segmentation")
    logger.info("  - step-annotation")
    logger.info("  - step-patch_extraction")
    logger.info("  - step-universal_training")
    logger.info("")
    logger.info("You can now run pipelines using:")
    logger.info("  controller.run()  # Runs as ClearML Pipeline")


if __name__ == "__main__":
    main()
