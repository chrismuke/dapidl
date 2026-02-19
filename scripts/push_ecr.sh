#!/usr/bin/env bash
# Build and push dapidl-agent Docker image to AWS ECR.
# Requires: aws sso login --profile admin
set -euo pipefail

ACCOUNT=273021981248
REGION=eu-central-1
REPO=dapidl-agent
ECR_URI=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO

echo "==> Creating ECR repository (if not exists)..."
aws ecr create-repository --repository-name "$REPO" --region "$REGION" --profile admin 2>/dev/null || true

echo "==> Logging in to ECR..."
aws ecr get-login-password --region "$REGION" --profile admin \
  | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

echo "==> Building image..."
docker build -f docker/Dockerfile.clearml-agent -t "$ECR_URI:latest" .

echo "==> Pushing image..."
docker push "$ECR_URI:latest"

echo "==> Done. Image: $ECR_URI:latest"
