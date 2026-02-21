#!/usr/bin/env python3
"""End-to-end test for ClearML AWS autoscaler.

Creates a task that:
1. Verifies S3 access by listing objects in the small lung-2fov dataset
2. Runs a minimal PyTorch training loop on GPU
3. Reports metrics to ClearML

Usage:
    uv run python scripts/test_autoscaler_e2e.py
"""

import os
import sys

# Only override config when running locally — the remote agent mounts its own config
if not os.environ.get("CLEARML_TASK_ID"):
    os.environ["CLEARML_CONFIG_FILE"] = os.path.expanduser("~/.clearml/clearml-chrism.conf")
    os.environ.pop("CLEARML_API_ACCESS_KEY", None)
    os.environ.pop("CLEARML_API_SECRET_KEY", None)

from clearml import Task

# Task.init() creates the task locally OR reconnects to existing task on remote agent
task = Task.init(
    project_name="DAPIDL/tests",
    task_name="e2e-autoscaler-s3-gpu-test",
    task_type=Task.TaskTypes.testing,
    auto_connect_frameworks=False,
    auto_connect_arg_parser=False,
)

if Task.running_locally():
    task.set_parameter("General/s3_test_path", "s3://dapidl/raw-data/xenium-lung-2fov/")
    task.set_parameter("General/epochs", 3)
    task.set_parameter("General/batch_size", 32)
    task.set_base_docker(
        docker_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        docker_arguments="--ipc=host --network host",
    )
    # Pin packages: boto3 for S3, skip torch upgrade (docker image has it)
    task.set_packages(["boto3", "torch==2.5.1", "clearml"])
    # Enqueue to gpu-training and exit locally
    task.execute_remotely(queue_name="gpu-training")
    # When running remotely, execute_remotely() is a no-op and continues below

# --- REMOTE: Run tests ---
logger = task.get_logger()
params = task.get_parameters_as_dict().get("General", {})
s3_path = params.get("s3_test_path", "s3://dapidl/raw-data/xenium-lung-2fov/")
epochs = int(params.get("epochs", 3))
batch_size = int(params.get("batch_size", 32))

# --- Test 1: S3 Access ---
print(f"\n=== Test 1: S3 Access ({s3_path}) ===")
import boto3

s3 = boto3.client("s3", region_name="eu-central-1")
bucket = s3_path.split("/")[2]
prefix = "/".join(s3_path.split("/")[3:])

response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=5)
contents = response.get("Contents", [])
print(f"Found {len(contents)} objects in {s3_path}")
for obj in contents:
    print(f"  {obj['Key']} ({obj['Size']:,} bytes)")

total_size = sum(obj["Size"] for obj in contents)
logger.report_scalar("s3", "objects_found", value=len(contents), iteration=0)
logger.report_scalar("s3", "total_bytes_listed", value=total_size, iteration=0)
print("S3 access: OK")

# --- Test 2: GPU Availability ---
print("\n=== Test 2: GPU Check ===")
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    logger.report_scalar("gpu", "memory_gb", value=gpu_mem, iteration=0)
else:
    print("WARNING: No GPU available, running on CPU")
    logger.report_scalar("gpu", "memory_gb", value=0, iteration=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Test 3: Minimal Training Loop ---
print(f"\n=== Test 3: Training ({epochs} epochs, batch_size={batch_size}) ===")
model = torch.nn.Sequential(
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 3),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    x = torch.randn(batch_size * 10, 128, device=device)
    y = torch.randint(0, 3, (batch_size * 10,), device=device)

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size]
        yb = y[i : i + batch_size]

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == yb).sum().item()
        total += len(yb)

    avg_loss = total_loss / (len(x) // batch_size)
    accuracy = correct / total
    logger.report_scalar("train", "loss", value=avg_loss, iteration=epoch)
    logger.report_scalar("train", "accuracy", value=accuracy, iteration=epoch)
    print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.3f}")

print("\n=== All Tests Passed ===")
logger.report_scalar("result", "success", value=1.0, iteration=0)
print("Done!")
