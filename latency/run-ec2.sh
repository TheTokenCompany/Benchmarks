#!/usr/bin/env bash
# Quick one-shot: spin up a US EC2 instance, run the latency benchmark, pull results back.
#
# Prerequisites:
#   - AWS CLI configured with credentials (aws configure)
#   - A key pair in the target region (or set KEY_NAME below)
#   - The .env file at ../benchmarks/.env with API keys
#
# Usage:
#   ./run-ec2.sh              # launch, run, fetch results, terminate
#   ./run-ec2.sh --keep       # same but don't terminate the instance at the end
#
# Env overrides:
#   REGION        AWS region          (default: us-east-1)
#   INSTANCE_TYPE Instance type       (default: t3.medium)
#   KEY_NAME      EC2 key pair name   (default: latency-bench)
#   KEY_FILE      Path to .pem file   (default: ~/.ssh/${KEY_NAME}.pem)

set -euo pipefail

REGION="${REGION:-us-west-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"
KEY_NAME="${KEY_NAME:-latency-bench}"
KEY_FILE="${KEY_FILE:-$HOME/.ssh/${KEY_NAME}.pem}"
KEEP=false
[[ "${1:-}" == "--keep" ]] && KEEP=true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$BENCH_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found. API keys are needed on the instance."
  exit 1
fi

echo "==> Region: $REGION | Instance: $INSTANCE_TYPE | Key: $KEY_NAME"

# --- Create key pair if it doesn't exist ---
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
  echo "==> Creating key pair '$KEY_NAME'..."
  aws ec2 create-key-pair --key-name "$KEY_NAME" --region "$REGION" \
    --query 'KeyMaterial' --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  echo "    Saved to $KEY_FILE"
fi

# --- Security group (allow SSH) ---
SG_NAME="latency-bench-sg"
SG_ID=$(aws ec2 describe-security-groups --group-names "$SG_NAME" --region "$REGION" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)

if [[ -z "$SG_ID" || "$SG_ID" == "None" ]]; then
  echo "==> Creating security group '$SG_NAME'..."
  SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" \
    --description "SSH for latency benchmark" --region "$REGION" \
    --query 'GroupId' --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
    --protocol tcp --port 22 --cidr 0.0.0.0/0
fi
echo "    SG: $SG_ID"

# --- Find latest Amazon Linux 2023 AMI ---
AMI_ID=$(aws ec2 describe-images --region "$REGION" \
  --owners amazon \
  --filters "Name=name,Values=al2023-ami-2023*-x86_64" "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)
echo "    AMI: $AMI_ID"

# --- Launch instance ---
echo "==> Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --count 1 \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=latency-bench}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "    Instance: $INSTANCE_ID"

cleanup() {
  if [[ "$KEEP" == false ]]; then
    echo "==> Terminating instance $INSTANCE_ID..."
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
    echo "    Done."
  else
    echo "==> Instance kept alive: $INSTANCE_ID (remember to terminate it later!)"
  fi
}
trap cleanup EXIT

# --- Wait for running + get public IP ---
echo "==> Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "    IP: $PUBLIC_IP"

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $KEY_FILE ec2-user@$PUBLIC_IP"
SCP="scp -o StrictHostKeyChecking=no -i $KEY_FILE"

# --- Wait for SSH ---
echo "==> Waiting for SSH..."
for i in $(seq 1 30); do
  if $SSH "echo ok" &>/dev/null; then break; fi
  sleep 5
done

# --- Install deps and upload code ---
echo "==> Setting up instance..."
$SSH "sudo dnf install -y python3.11 python3.11-pip git &>/dev/null"

# Upload benchmark files
$SCP "$SCRIPT_DIR/benchmark.py" "ec2-user@$PUBLIC_IP:~/benchmark.py"
$SCP "$SCRIPT_DIR/inputs.json" "ec2-user@$PUBLIC_IP:~/inputs.json"
$SCP "$ENV_FILE" "ec2-user@$PUBLIC_IP:~/.env"

# Install Python deps
$SSH "python3.11 -m pip install --quiet httpx python-dotenv"

# Patch the benchmark to load .env from home dir
$SSH 'sed -i "s|HERE.parent / \".env\"|Path.home() / \".env\"|" ~/benchmark.py'

echo "==> Running benchmark (50 runs × 60 combos = 3000 requests, this will take a while)..."
$SSH "cd ~ && python3.11 benchmark.py" 2>&1 | tee "$SCRIPT_DIR/ec2-output.log"

# --- Pull results ---
echo "==> Fetching results..."
$SCP "ec2-user@$PUBLIC_IP:~/results.json" "$SCRIPT_DIR/results-ec2-${REGION}.json"
echo "    Saved to: $SCRIPT_DIR/results-ec2-${REGION}.json"

echo "==> Done!"
