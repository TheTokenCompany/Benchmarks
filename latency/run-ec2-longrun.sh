#!/usr/bin/env bash
# Launch a long-running EC2 instance that benchmarks TTC gateway latency
# every 15 minutes for a week+.
#
# - Secured: SSH only from your current public IP
# - Runs longrun.py via cron every 15 min
# - Logs to ~/longrun-results.jsonl (JSONL, one line per request)
# - Cron output logged to ~/longrun-cron.log
#
# Usage:
#   ./run-ec2-longrun.sh          # launch and set up
#   ./run-ec2-longrun.sh fetch    # fetch results from running instance
#   ./run-ec2-longrun.sh stop     # terminate the instance
#
# The instance ID is saved to .longrun-instance for later fetch/stop.

set -euo pipefail

REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.micro}"
KEY_NAME="${KEY_NAME:-latency-bench}"
KEY_FILE="${KEY_FILE:-$HOME/.ssh/${KEY_NAME}.pem}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$BENCH_DIR/.env"
INSTANCE_FILE="$SCRIPT_DIR/.longrun-instance"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $KEY_FILE"

get_instance_ip() {
  local iid="$1"
  aws ec2 describe-instances --instance-ids "$iid" --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

# --- fetch: pull results from running instance ---
if [[ "${1:-}" == "fetch" ]]; then
  if [[ ! -f "$INSTANCE_FILE" ]]; then
    echo "ERROR: No running instance found ($INSTANCE_FILE missing)"
    exit 1
  fi
  INSTANCE_ID=$(cat "$INSTANCE_FILE")
  IP=$(get_instance_ip "$INSTANCE_ID")
  echo "==> Fetching results from $INSTANCE_ID ($IP)..."
  scp $SSH_OPTS "ec2-user@$IP:~/longrun-results.jsonl" "$SCRIPT_DIR/longrun-results.jsonl"
  scp $SSH_OPTS "ec2-user@$IP:~/longrun-cron.log" "$SCRIPT_DIR/longrun-cron.log" 2>/dev/null || true
  echo "    Saved to $SCRIPT_DIR/longrun-results.jsonl"
  # Quick stats
  TOTAL=$(wc -l < "$SCRIPT_DIR/longrun-results.jsonl" | tr -d ' ')
  OK=$(grep -c '"success": true' "$SCRIPT_DIR/longrun-results.jsonl" || true)
  FAIL=$((TOTAL - OK))
  FIRST=$(head -1 "$SCRIPT_DIR/longrun-results.jsonl" | python3 -c "import sys,json; print(json.load(sys.stdin)['timestamp'][:16])" 2>/dev/null || echo "?")
  LAST=$(tail -1 "$SCRIPT_DIR/longrun-results.jsonl" | python3 -c "import sys,json; print(json.load(sys.stdin)['timestamp'][:16])" 2>/dev/null || echo "?")
  echo "    Total: $TOTAL requests ($OK ok, $FAIL failed)"
  echo "    Range: $FIRST → $LAST"
  exit 0
fi

# --- stop: terminate the instance ---
if [[ "${1:-}" == "stop" ]]; then
  if [[ ! -f "$INSTANCE_FILE" ]]; then
    echo "No instance file found."
    exit 0
  fi
  INSTANCE_ID=$(cat "$INSTANCE_FILE")
  echo "==> Terminating $INSTANCE_ID..."
  aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
  rm -f "$INSTANCE_FILE"
  echo "    Done."
  exit 0
fi

# --- Main: launch and set up ---
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found."
  exit 1
fi

# Get current public IP for security group
MY_IP=$(curl -s https://checkip.amazonaws.com | tr -d '\n')
echo "==> Your IP: $MY_IP"
echo "==> Region: $REGION | Instance: $INSTANCE_TYPE"

# --- Key pair ---
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
  echo "==> Creating key pair '$KEY_NAME'..."
  aws ec2 create-key-pair --key-name "$KEY_NAME" --region "$REGION" \
    --query 'KeyMaterial' --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
fi

# --- Security group (SSH from current IP only) ---
SG_NAME="latency-longrun-sg"
SG_ID=$(aws ec2 describe-security-groups --group-names "$SG_NAME" --region "$REGION" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)

if [[ -z "$SG_ID" || "$SG_ID" == "None" ]]; then
  echo "==> Creating security group '$SG_NAME' (SSH from $MY_IP only)..."
  SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" \
    --description "SSH for longrun latency benchmark - restricted" --region "$REGION" \
    --query 'GroupId' --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
    --protocol tcp --port 22 --cidr "$MY_IP/32"
else
  # Update the IP rule in case it changed
  echo "==> Updating security group to allow SSH from $MY_IP..."
  # Revoke all existing SSH rules and add current IP
  aws ec2 revoke-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
    --protocol tcp --port 22 --cidr "0.0.0.0/0" 2>/dev/null || true
  # Try to add (may already exist)
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
    --protocol tcp --port 22 --cidr "$MY_IP/32" 2>/dev/null || true
fi
echo "    SG: $SG_ID"

# --- AMI ---
AMI_ID=$(aws ec2 describe-images --region "$REGION" \
  --owners amazon \
  --filters "Name=name,Values=al2023-ami-2023*-x86_64" "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)
echo "    AMI: $AMI_ID"

# --- Launch ---
echo "==> Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --count 1 \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=latency-longrun}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "    Instance: $INSTANCE_ID"
echo "$INSTANCE_ID" > "$INSTANCE_FILE"

# --- Wait ---
echo "==> Waiting for instance..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
PUBLIC_IP=$(get_instance_ip "$INSTANCE_ID")
echo "    IP: $PUBLIC_IP"

SSH="ssh $SSH_OPTS ec2-user@$PUBLIC_IP"
SCP="scp $SSH_OPTS"

echo "==> Waiting for SSH..."
for i in $(seq 1 30); do
  if $SSH "echo ok" &>/dev/null; then break; fi
  sleep 5
done

# --- Setup ---
echo "==> Installing deps..."
$SSH "sudo dnf install -y python3.11 python3.11-pip &>/dev/null"

echo "==> Uploading files..."
$SCP "$SCRIPT_DIR/longrun.py" "ec2-user@$PUBLIC_IP:~/longrun.py"
$SCP "$SCRIPT_DIR/inputs.json" "ec2-user@$PUBLIC_IP:~/inputs.json"
$SCP "$ENV_FILE" "ec2-user@$PUBLIC_IP:~/.env"

$SSH "python3.11 -m pip install --quiet httpx python-dotenv"

# --- Verify it works ---
echo "==> Running initial test..."
$SSH "cd ~ && python3.11 longrun.py"

# --- Set up systemd timer (cron not available on AL2023) ---
echo "==> Setting up systemd timer (every 15 min)..."
$SSH 'sudo tee /etc/systemd/system/longrun-bench.service > /dev/null << EOSVC
[Unit]
Description=TTC Gateway latency benchmark

[Service]
Type=oneshot
User=ec2-user
WorkingDirectory=/home/ec2-user
ExecStart=/usr/bin/python3.11 /home/ec2-user/longrun.py
StandardOutput=append:/home/ec2-user/longrun-cron.log
StandardError=append:/home/ec2-user/longrun-cron.log
EOSVC'

$SSH 'sudo tee /etc/systemd/system/longrun-bench.timer > /dev/null << EOTMR
[Unit]
Description=Run TTC latency benchmark every 15 minutes

[Timer]
OnBootSec=1min
OnUnitActiveSec=15min
AccuracySec=1s

[Install]
WantedBy=timers.target
EOTMR'

$SSH "sudo systemctl daemon-reload && sudo systemctl enable --now longrun-bench.timer"

# Verify
$SSH "sudo systemctl status longrun-bench.timer --no-pager"

echo ""
echo "=========================================="
echo "Long-running benchmark is live!"
echo "=========================================="
echo "  Instance: $INSTANCE_ID"
echo "  IP:       $PUBLIC_IP"
echo "  SSH:      ssh $SSH_OPTS ec2-user@$PUBLIC_IP"
echo ""
echo "  Runs every 15 min, testing gpt-5-mini + claude-sonnet-4.6 via TTC gateway"
echo "  Rotates through compression: none → low → high"
echo ""
echo "  Fetch results:  ./run-ec2-longrun.sh fetch"
echo "  Stop & terminate: ./run-ec2-longrun.sh stop"
echo "=========================================="
