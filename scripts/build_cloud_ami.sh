#!/usr/bin/env bash
# Build a custom AMI for ClearML cloud agents with pre-loaded Docker image.
#
# This creates an AMI from the AWS Deep Learning Base AMI with the
# dapidl-agent:latest Docker image pre-loaded, eliminating the need for
# ECR authentication at agent boot time.
#
# Prerequisites:
#   - aws sso login --profile admin
#   - dapidl-agent:latest Docker image built locally
#     (docker build -f docker/Dockerfile.clearml-agent -t dapidl-agent:latest .)
#
# Usage:
#   ./scripts/build_cloud_ami.sh
#
# The script will:
#   1. Save dapidl-agent:latest to a tarball
#   2. Launch a temporary EC2 instance from the Deep Learning AMI
#   3. Upload and load the Docker image
#   4. Create a new AMI from the instance
#   5. Update the ClearML autoscaler config with the new AMI ID
#   6. Terminate the temporary instance
set -euo pipefail

# Configuration
REGION="eu-central-1"
PROFILE="${AWS_PROFILE:-dapidl}"
BASE_AMI="ami-0d9cfda3e4350f2c3"  # Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
INSTANCE_TYPE="g6.xlarge"
KEY_NAME="clearml-agent-key"
SECURITY_GROUP="sg-0bc7bb543984718bc"
SUBNET="subnet-0b18309028c8a2c73"
IMAGE_NAME="dapidl-agent"
IMAGE_TAG="latest"
AMI_PREFIX="dapidl-cloud-agent"

echo "=== Step 1: Save Docker image to tarball ==="
TMPDIR=$(mktemp -d)
TARBALL="$TMPDIR/dapidl-agent.tar.gz"
echo "Saving $IMAGE_NAME:$IMAGE_TAG to $TARBALL ..."
docker save "$IMAGE_NAME:$IMAGE_TAG" | gzip > "$TARBALL"
SIZE_GB=$(du -sh "$TARBALL" | cut -f1)
echo "Image saved: $SIZE_GB"

echo ""
echo "=== Step 2: Launch temporary EC2 instance ==="
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$BASE_AMI" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --subnet-id "$SUBNET" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":120,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ami-builder-temp}]" \
    --region "$REGION" \
    --profile "$PROFILE" \
    --query 'Instances[0].InstanceId' \
    --output text)
echo "Launched instance: $INSTANCE_ID"

cleanup() {
    echo "Terminating temporary instance $INSTANCE_ID ..."
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" --profile "$PROFILE" > /dev/null 2>&1 || true
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION" --profile "$PROFILE"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --profile "$PROFILE" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)
echo "Instance IP: $PUBLIC_IP"

echo "Waiting for SSH to become available..."
for i in $(seq 1 30); do
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i ~/.ssh/clearml-agent-key.pem ubuntu@"$PUBLIC_IP" "echo ready" 2>/dev/null && break
    echo "  attempt $i..."
    sleep 10
done

echo ""
echo "=== Step 3: Upload and load Docker image ==="
echo "Uploading $SIZE_GB tarball (this may take a while)..."
scp -o StrictHostKeyChecking=no -i ~/.ssh/clearml-agent-key.pem "$TARBALL" ubuntu@"$PUBLIC_IP":/tmp/dapidl-agent.tar.gz

echo "Loading Docker image on instance..."
ssh -o StrictHostKeyChecking=no -i ~/.ssh/clearml-agent-key.pem ubuntu@"$PUBLIC_IP" \
    "sudo docker load < /tmp/dapidl-agent.tar.gz && sudo rm /tmp/dapidl-agent.tar.gz && sudo docker images"

echo ""
echo "=== Step 4: Create AMI ==="
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
AMI_NAME="${AMI_PREFIX}-${TIMESTAMP}"

echo "Stopping instance for clean snapshot..."
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" --profile "$PROFILE" > /dev/null
aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION" --profile "$PROFILE"

echo "Creating AMI: $AMI_NAME"
NEW_AMI_ID=$(aws ec2 create-image \
    --instance-id "$INSTANCE_ID" \
    --name "$AMI_NAME" \
    --description "DAPIDL cloud agent: Deep Learning AMI + dapidl-agent:latest Docker image pre-loaded" \
    --region "$REGION" \
    --profile "$PROFILE" \
    --query 'ImageId' \
    --output text)
echo "AMI creation started: $NEW_AMI_ID"

echo "Waiting for AMI to become available (this takes 5-15 minutes)..."
aws ec2 wait image-available --image-ids "$NEW_AMI_ID" --region "$REGION" --profile "$PROFILE"
echo "AMI ready: $NEW_AMI_ID"

echo ""
echo "=== Step 5: Summary ==="
echo "  New AMI ID:  $NEW_AMI_ID"
echo "  AMI Name:    $AMI_NAME"
echo "  Base AMI:    $BASE_AMI"
echo "  Region:      $REGION"
echo ""
echo "To update the ClearML autoscaler, change ami_id in the"
echo "General configuration of task 9328c6ff24bc4b88b34b83fbd2b68a4c"
echo "from '$BASE_AMI' to '$NEW_AMI_ID'"
echo ""
echo "The autoscaler's extra_vm_bash_script no longer needs ECR login."
echo "It only needs ClearML env overrides."
