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
echo "=== Step 3b: Bake ClearML agent config with public URLs ==="
# The ClearML autoscaler generates a boot script that exports CLEARML_API_HOST
# using its own internal address (http://10.0.0.106:8008). EC2 instances can't
# reach this. The extra_vm_bash_script is supposed to override, but it's fragile.
# Baking the correct config into the AMI ensures the agent always connects via
# the public URL, regardless of env var ordering issues.
ssh -o StrictHostKeyChecking=no -i ~/.ssh/clearml-agent-key.pem ubuntu@"$PUBLIC_IP" \
    "sudo tee /root/clearml.conf > /dev/null" <<'CLEARML_CONF'
api {
    web_server: https://clearml.chrism.io
    api_server: https://api.clearml.chrism.io
    files_server: https://files.clearml.chrism.io
    credentials {
        # Overridden by env vars at boot time
        access_key: ""
        secret_key: ""
    }
}
agent {
    docker_force_pull: false
    docker_args: "--network host --ipc=host -e WANDB_MODE=disabled"
    package_manager {
        type: pip
        system_site_packages: true
    }
}
CLEARML_CONF

# Install clearml-agent into the AMI so boot is faster (no pip install needed)
ssh -o StrictHostKeyChecking=no -i ~/.ssh/clearml-agent-key.pem ubuntu@"$PUBLIC_IP" \
    "sudo python3 -m virtualenv /clearml_agent_venv && sudo /clearml_agent_venv/bin/pip install clearml-agent"

echo ""
echo "=== Step 3c: Pre-install CloudWatch Agent and GPU monitoring ==="
# Pre-installing CloudWatch Agent saves ~30s at boot time. The cron fallback
# script is also baked in so GPU metrics push immediately after boot without
# needing to curl the setup script from GitHub.
ssh -o StrictHostKeyChecking=no -i ~/.ssh/clearml-agent-key.pem ubuntu@"$PUBLIC_IP" << 'GPU_MONITOR'
set -e
echo "[AMI] Installing CloudWatch Agent..."
wget -q https://amazoncloudwatch-agent-eu-central-1.s3.eu-central-1.amazonaws.com/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb
rm -f amazon-cloudwatch-agent.deb

echo "[AMI] Writing CloudWatch Agent GPU config..."
sudo tee /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json > /dev/null << 'CWCONFIG'
{
  "metrics": {
    "namespace": "GPU",
    "metrics_collected": {
      "nvidia_gpu": {
        "measurement": [
          "utilization_gpu",
          "utilization_memory",
          "memory_total",
          "memory_used",
          "memory_free",
          "temperature_gpu",
          "power_draw"
        ],
        "metrics_collection_interval": 30
      }
    },
    "append_dimensions": {
      "InstanceId": "${aws:InstanceId}",
      "InstanceType": "${aws:InstanceType}"
    }
  }
}
CWCONFIG

echo "[AMI] Writing cron fallback GPU metrics script..."
sudo tee /usr/local/bin/gpu-metrics-push.sh > /dev/null << 'CRONSCRIPT'
#!/bin/bash
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
read GPU_UTIL MEM_UTIL MEM_USED MEM_TOTAL TEMP POWER <<< $(nvidia-smi \
  --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader,nounits | tr ',' ' ')
aws cloudwatch put-metric-data --region "$REGION" --namespace "GPU" --metric-data \
  "[{\"MetricName\":\"utilization_gpu\",\"Value\":$GPU_UTIL,\"Unit\":\"Percent\",\"Dimensions\":[{\"Name\":\"InstanceId\",\"Value\":\"$INSTANCE_ID\"},{\"Name\":\"InstanceType\",\"Value\":\"$INSTANCE_TYPE\"}]},{\"MetricName\":\"utilization_memory\",\"Value\":$MEM_UTIL,\"Unit\":\"Percent\",\"Dimensions\":[{\"Name\":\"InstanceId\",\"Value\":\"$INSTANCE_ID\"},{\"Name\":\"InstanceType\",\"Value\":\"$INSTANCE_TYPE\"}]},{\"MetricName\":\"memory_used\",\"Value\":$MEM_USED,\"Unit\":\"Megabytes\",\"Dimensions\":[{\"Name\":\"InstanceId\",\"Value\":\"$INSTANCE_ID\"},{\"Name\":\"InstanceType\",\"Value\":\"$INSTANCE_TYPE\"}]},{\"MetricName\":\"memory_total\",\"Value\":$MEM_TOTAL,\"Unit\":\"Megabytes\",\"Dimensions\":[{\"Name\":\"InstanceId\",\"Value\":\"$INSTANCE_ID\"},{\"Name\":\"InstanceType\",\"Value\":\"$INSTANCE_TYPE\"}]},{\"MetricName\":\"temperature_gpu\",\"Value\":$TEMP,\"Unit\":\"None\",\"Dimensions\":[{\"Name\":\"InstanceId\",\"Value\":\"$INSTANCE_ID\"},{\"Name\":\"InstanceType\",\"Value\":\"$INSTANCE_TYPE\"}]},{\"MetricName\":\"power_draw\",\"Value\":$POWER,\"Unit\":\"None\",\"Dimensions\":[{\"Name\":\"InstanceId\",\"Value\":\"$INSTANCE_ID\"},{\"Name\":\"InstanceType\",\"Value\":\"$INSTANCE_TYPE\"}]}]"
CRONSCRIPT
sudo chmod +x /usr/local/bin/gpu-metrics-push.sh
echo "[AMI] CloudWatch Agent and GPU monitoring pre-installed."
GPU_MONITOR

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
