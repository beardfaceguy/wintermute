#!/usr/bin/env bash
#
# Launch a vLLM serving instance on AWS EC2 and start inference.
#
# Usage:
#   ./scripts/aws_commands/vllm_serve.sh launch    # launch instance + start vLLM
#   ./scripts/aws_commands/vllm_serve.sh status     # check instance state + IP
#   ./scripts/aws_commands/vllm_serve.sh stop        # stop instance (preserves EBS)
#   ./scripts/aws_commands/vllm_serve.sh terminate   # terminate instance
#
# After launch, set the env var for wintermute:
#   export VLLM_HOST=<public_ip>
#
# All infrastructure config is read from config/shared_api_config.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$REPO_ROOT/config/shared_api_config.json"

export AWS_PROFILE="${AWS_PROFILE:-experimental-admin}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

# Read AWS config from shared config
read_config() {
    python3 -c "
import json, sys
with open('$CONFIG_FILE') as f:
    cfg = json.load(f)['vllm']['aws']
print(cfg.get(sys.argv[1], ''))
" "$1"
}

INSTANCE_TYPE="$(read_config instance_type)"
AMI="$(read_config ami)"
SG="$(read_config security_group)"
KEY_PAIR="$(read_config key_pair)"
INSTANCE_PROFILE="$(read_config instance_profile)"
ECR_IMAGE="$(read_config ecr_image)"
HF_EXPORT_S3="$(read_config hf_export_s3)"
SERVE_PORT="$(read_config serve_port)"
GPU_MEM_UTIL="$(read_config gpu_memory_utilization)"
ACCOUNT_ID="$(read_config account_id)"
REGION="$(read_config region)"

TAG_SPEC="ResourceType=instance,Tags=[
  {Key=Owner,Value=patrick.clawson},
  {Key=Project,Value=Wintermute},
  {Key=Env,Value=staging},
  {Key=CostCenter,Value=ai-ml-training},
  {Key=Purpose,Value=vllm-inference},
  {Key=Name,Value=wintermute-vllm-serve}
]"

# Find existing wintermute-vllm-serve instance
find_instance() {
    aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=wintermute-vllm-serve" \
                  "Name=instance-state-name,Values=running,stopped,pending" \
        --query 'Reservations[].Instances[0].[InstanceId,State.Name,PublicIpAddress]' \
        --output text --no-cli-pager 2>/dev/null | head -1
}

cmd_launch() {
    echo "=== Launching vLLM serving instance ==="
    echo "  Instance type: $INSTANCE_TYPE"
    echo "  AMI:           $AMI"
    echo "  ECR image:     $ECR_IMAGE"
    echo "  Model S3:      $HF_EXPORT_S3"
    echo "  Serve port:    $SERVE_PORT"
    echo ""

    existing="$(find_instance)"
    if [ -n "$existing" ]; then
        iid=$(echo "$existing" | awk '{print $1}')
        state=$(echo "$existing" | awk '{print $2}')
        ip=$(echo "$existing" | awk '{print $3}')
        echo "Existing instance found: $iid ($state, IP: $ip)"
        if [ "$state" = "stopped" ]; then
            echo "Starting stopped instance..."
            aws ec2 start-instances --instance-ids "$iid" --no-cli-pager
            echo "Waiting for instance to reach running state..."
            aws ec2 wait instance-running --instance-ids "$iid" --no-cli-pager
            ip=$(aws ec2 describe-instances --instance-ids "$iid" \
                --query 'Reservations[0].Instances[0].PublicIpAddress' \
                --output text --no-cli-pager)
            echo "Instance running at: $ip"
        elif [ "$state" = "running" ]; then
            echo "Instance already running."
        fi
        echo ""
        echo "export VLLM_HOST=$ip"
        return
    fi

    echo "Launching new on-demand instance..."
    iid=$(aws ec2 run-instances \
        --image-id "$AMI" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_PAIR" \
        --security-group-ids "$SG" \
        --iam-instance-profile "Name=$INSTANCE_PROFILE" \
        --block-device-mappings '[{"DeviceName":"/dev/sdf","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
        --tag-specifications "$TAG_SPEC" \
        --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
        --query 'Instances[0].InstanceId' \
        --output text --no-cli-pager)

    echo "Instance launched: $iid"
    echo "Waiting for running state..."
    aws ec2 wait instance-running --instance-ids "$iid" --no-cli-pager

    ip=$(aws ec2 describe-instances --instance-ids "$iid" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text --no-cli-pager)
    echo "Instance running at: $ip"

    echo ""
    echo "Waiting for SSM agent to come online..."
    for i in $(seq 1 30); do
        ssm_status=$(aws ssm describe-instance-information \
            --filters "Key=InstanceIds,Values=$iid" \
            --query 'InstanceInformationList[0].PingStatus' \
            --output text --no-cli-pager 2>/dev/null || echo "None")
        if [ "$ssm_status" = "Online" ]; then
            echo "SSM agent online."
            break
        fi
        sleep 10
    done

    echo ""
    echo "Starting vLLM via SSM..."
    cmd_id=$(aws ssm send-command \
        --instance-ids "$iid" \
        --document-name "AWS-RunShellScript" \
        --parameters "{
            \"commands\": [
                \"set -ex\",
                \"mkdir -p /mnt/data/hf_model\",
                \"aws s3 sync $HF_EXPORT_S3 /mnt/data/hf_model/ --only-show-errors\",
                \"aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com\",
                \"docker pull $ECR_IMAGE\",
                \"docker rm -f vllm-serve 2>/dev/null || true\",
                \"docker run -d --gpus all --name vllm-serve -p $SERVE_PORT:8000 -v /mnt/data/hf_model:/model $ECR_IMAGE --model /model --dtype float16 --port 8000 --gpu-memory-utilization $GPU_MEM_UTIL\",
                \"echo 'vLLM container started on port $SERVE_PORT'\",
                \"sleep 120\",
                \"curl -sf http://localhost:$SERVE_PORT/v1/models || echo 'WARNING: vLLM not yet ready after 120s warmup'\"
            ],
            \"executionTimeout\": [\"600\"]
        }" \
        --timeout-seconds 600 \
        --query 'Command.CommandId' \
        --output text --no-cli-pager)

    echo "SSM command: $cmd_id"
    echo "Monitoring startup (this takes ~2-3 minutes for GPU warmup)..."

    for i in $(seq 1 30); do
        sleep 15
        job_status=$(aws ssm list-command-invocations \
            --command-id "$cmd_id" \
            --no-cli-pager \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['CommandInvocations'][0]['Status'] if d['CommandInvocations'] else 'Pending')" 2>/dev/null || echo "Pending")
        echo "  [$i] Status: $job_status"
        if [ "$job_status" = "Success" ]; then
            echo ""
            echo "=== vLLM is ready ==="
            echo "  Endpoint: http://$ip:$SERVE_PORT/v1"
            echo "  Models:   curl http://$ip:$SERVE_PORT/v1/models"
            echo ""
            echo "Set the env var:"
            echo "  export VLLM_HOST=$ip"
            return
        elif [ "$job_status" = "Failed" ] || [ "$job_status" = "TimedOut" ]; then
            echo "SSM command failed ($job_status). Check logs:"
            echo "  aws ssm get-command-invocation --command-id $cmd_id --instance-id $iid --no-cli-pager"
            return 1
        fi
    done

    echo ""
    echo "Startup still in progress. Check manually:"
    echo "  curl http://$ip:$SERVE_PORT/v1/models"
    echo "  export VLLM_HOST=$ip"
}

cmd_status() {
    echo "=== vLLM instance status ==="
    existing="$(find_instance)"
    if [ -z "$existing" ]; then
        echo "No wintermute-vllm-serve instance found."
        return
    fi
    iid=$(echo "$existing" | awk '{print $1}')
    state=$(echo "$existing" | awk '{print $2}')
    ip=$(echo "$existing" | awk '{print $3}')
    echo "  Instance: $iid"
    echo "  State:    $state"
    echo "  IP:       ${ip:-N/A}"
    if [ "$state" = "running" ] && [ "$ip" != "None" ] && [ -n "$ip" ]; then
        echo ""
        echo "  export VLLM_HOST=$ip"
        echo ""
        echo "Checking vLLM health..."
        curl -sf --connect-timeout 5 "http://$ip:$SERVE_PORT/v1/models" 2>/dev/null \
            && echo "" \
            || echo "  vLLM not responding (may still be starting up)"
    fi
}

cmd_stop() {
    existing="$(find_instance)"
    if [ -z "$existing" ]; then
        echo "No instance to stop."
        return
    fi
    iid=$(echo "$existing" | awk '{print $1}')
    echo "Stopping $iid..."
    aws ec2 stop-instances --instance-ids "$iid" --no-cli-pager
    echo "Instance stopping. EBS data preserved."
}

cmd_terminate() {
    existing="$(find_instance)"
    if [ -z "$existing" ]; then
        echo "No instance to terminate."
        return
    fi
    iid=$(echo "$existing" | awk '{print $1}')
    echo "Terminating $iid..."
    aws ec2 terminate-instances --instance-ids "$iid" --no-cli-pager
    echo "Instance terminating. Remember to unset VLLM_HOST."
}

case "${1:-help}" in
    launch)    cmd_launch ;;
    status)    cmd_status ;;
    stop)      cmd_stop ;;
    terminate) cmd_terminate ;;
    *)
        echo "Usage: $0 {launch|status|stop|terminate}"
        echo ""
        echo "  launch     Launch EC2 + start vLLM (or restart stopped instance)"
        echo "  status     Show instance state, IP, and vLLM health"
        echo "  stop       Stop instance (preserves EBS, saves cost)"
        echo "  terminate  Terminate instance (destroys everything)"
        ;;
esac
