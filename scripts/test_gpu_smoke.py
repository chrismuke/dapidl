"""Tiny GPU smoke test for ClearML agent verification.

Creates a ClearML task that:
1. Checks GPU availability
2. Runs a small matrix multiply on GPU
3. Reports GPU memory usage
4. Sleeps briefly to simulate work
"""
from __future__ import annotations

import time

from clearml import Task


def main() -> None:
    task = Task.init(project_name="DAPIDL/tests", task_name="gpu-smoke-test")
    logger = task.get_logger()

    try:
        import torch

        if not torch.cuda.is_available():
            logger.report_text("NO GPU AVAILABLE")
            task.mark_failed(status_reason="No GPU found")
            return

        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        logger.report_text(f"GPU: {name} ({total} MB)")

        # Small matmul to exercise the GPU
        a = torch.randn(2048, 2048, device=device)
        b = torch.randn(2048, 2048, device=device)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(20):
            c = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        used = torch.cuda.memory_allocated(0) // (1024**2)
        logger.report_scalar("gpu", "memory_used_mb", used, 0)
        logger.report_scalar("gpu", "matmul_time_s", elapsed, 0)
        logger.report_text(f"20x matmul(2048x2048) in {elapsed:.3f}s, {used} MB used")

        # Brief sleep to keep the task visible in monitoring
        time.sleep(30)
        logger.report_text("Done!")

    except Exception as e:
        logger.report_text(f"Error: {e}")
        task.mark_failed(status_reason=str(e))
        return

    task.mark_completed()


if __name__ == "__main__":
    main()
