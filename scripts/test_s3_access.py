"""S3 access test for ClearML cloud agent verification.

Tests that IAM role-based S3 access works from inside Docker containers
on autoscaler-provisioned EC2 instances.
"""
import os
import time

from clearml import Task


def main() -> None:
    task = Task.init(project_name="DAPIDL/tests", task_name="s3-access-test")
    logger = task.get_logger()

    try:
        import boto3

        s3 = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "eu-central-1"),
        )

        # List objects in dapidl bucket
        resp = s3.list_objects_v2(Bucket="dapidl", Prefix="raw-data/", MaxKeys=5)
        keys = [o["Key"] for o in resp.get("Contents", [])]
        logger.report_text(f"Listed {len(keys)} objects: {keys}")

        # Write test object
        test_key = "test/autoscaler-robustness/s3-test.txt"
        s3.put_object(Bucket="dapidl", Key=test_key, Body=b"autoscaler s3 test OK")
        logger.report_text(f"Wrote {test_key}")

        # Read it back
        obj = s3.get_object(Bucket="dapidl", Key=test_key)
        body = obj["Body"].read().decode()
        assert body == "autoscaler s3 test OK", f"Mismatch: {body}"
        logger.report_text(f"Read back: {body}")

        # Delete it
        s3.delete_object(Bucket="dapidl", Key=test_key)
        logger.report_text(f"Deleted {test_key}")

        logger.report_text("S3 test PASSED")
        time.sleep(5)

    except Exception as e:
        logger.report_text(f"S3 test FAILED: {e}")
        task.mark_failed(status_reason=str(e))
        return

    task.mark_completed()


if __name__ == "__main__":
    main()
