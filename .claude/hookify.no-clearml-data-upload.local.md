---
name: no-clearml-data-upload
enabled: true
event: file
pattern: .*\.py$
action: warn
---

## Never Upload Large Data Directly to ClearML

**CRITICAL RULE**: Never save raw datasets, models, or generated datasets directly to ClearML file storage.

### What's Allowed on ClearML:
- Metrics and scalars
- Small configuration files
- Task parameters
- References/links to S3 locations

### What Must Go to S3:
- Raw datasets (Xenium, MERSCOPE outputs)
- Trained models (.ckpt, .pt files)
- Generated datasets (LMDB, processed data)
- Large artifacts (anything > 1MB)

### Correct Pattern:
```python
# WRONG - Direct upload to ClearML
task.upload_artifact("model", model_path)
task.upload_artifact("dataset", dataset_path)

# CORRECT - Upload to S3, register with ClearML
s3_path = upload_to_s3(local_path, "s3://dapidl/artifacts/...")
task.upload_artifact("model_s3_uri", s3_path)  # Just the URI string
# OR
task.register_artifact("model", s3_path)  # S3 reference
```

### S3 Configuration:
- Endpoint: `https://s3.eu-central-2.idrivee2.com`
- Bucket: `dapidl`
- Use boto3 or aws cli for uploads

### Why This Matters:
1. ClearML hosted storage has quota limits (already exhausted)
2. S3 is cheaper and more reliable for large files
3. Local paths don't work when running on remote agents
4. Prevents accidental 20GB+ uploads that break pipelines
