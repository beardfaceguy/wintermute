#!/usr/bin/env bash
#
# Launch / manage the Code Review model inference server on AWS EC2.
#
# Usage:
#   ./scripts/aws_commands/codereview_serve.sh launch    # launch instance + start server
#   ./scripts/aws_commands/codereview_serve.sh status     # check instance state + health
#   ./scripts/aws_commands/codereview_serve.sh stop       # stop instance (preserves EBS)
#   ./scripts/aws_commands/codereview_serve.sh terminate  # terminate instance
#
# After launch, set the env var for agents:
#   export CODEREVIEW_HOST=<public_ip>
#
# All infrastructure config is read from config/shared_api_config.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$REPO_ROOT/config/shared_api_config.json"

export AWS_PROFILE="${AWS_PROFILE:-experimental-admin}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

read_config() {
    python3 -c "
import json, sys
with open('$CONFIG_FILE') as f:
    cfg = json.load(f)['codereview']['aws']
print(cfg.get(sys.argv[1], ''))
" "$1"
}

INSTANCE_TYPE="$(read_config instance_type)"
AMI="$(read_config ami)"
SG="$(read_config security_group)"
KEY_PAIR="$(read_config key_pair)"
INSTANCE_PROFILE="$(read_config instance_profile)"
CKPT_S3="$(read_config ckpt_s3)"
TOKENIZER_S3="$(read_config tokenizer_s3)"
CODE_BUNDLE_S3="$(read_config code_bundle_s3)"
SERVE_PORT="$(read_config serve_port)"
REGION="$(read_config region)"

TAG_SPEC="ResourceType=instance,Tags=[
  {Key=Owner,Value=patrick.clawson},
  {Key=Project,Value=Wintermute},
  {Key=Env,Value=staging},
  {Key=CostCenter,Value=ai-ml-training},
  {Key=Purpose,Value=codereview-inference},
  {Key=Name,Value=wintermute-codereview-serve}
]"

find_instance() {
    aws ec2 describe-instances \
        --filters "Name=tag:Name,Values=wintermute-codereview-serve" \
                  "Name=instance-state-name,Values=running,stopped,pending" \
        --query 'Reservations[].Instances[0].[InstanceId,State.Name,PublicIpAddress]' \
        --output text --no-cli-pager 2>/dev/null | head -1
}

cmd_launch() {
    echo "=== Launching Code Review inference server ==="
    echo "  Instance type: $INSTANCE_TYPE"
    echo "  AMI:           $AMI"
    echo "  Checkpoint:    $CKPT_S3"
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
    else
        echo "Launching new on-demand instance..."
        iid=$(aws ec2 run-instances \
            --image-id "$AMI" \
            --instance-type "$INSTANCE_TYPE" \
            --key-name "$KEY_PAIR" \
            --security-group-ids "$SG" \
            --iam-instance-profile "Name=$INSTANCE_PROFILE" \
            --block-device-mappings '[{"DeviceName":"/dev/sdf","Ebs":{"VolumeSize":50,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
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
    fi

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
    echo "Deploying code review server via SSM..."
    cmd_id=$(aws ssm send-command \
        --instance-ids "$iid" \
        --document-name "AWS-RunShellScript" \
        --timeout-seconds 1800 \
        --cloud-watch-output-config "CloudWatchLogGroupName=/aws/ssm/titan-llm-training,CloudWatchOutputEnabled=true" \
        --parameters "{
            \"commands\": [
                \"set -ex\",
                \"DATA_ROOT=/opt/dlami/nvme\",
                \"if ! mountpoint -q /opt/dlami/nvme 2>/dev/null; then DATA_ROOT=/mnt/data; mkdir -p /mnt/data; fi\",
                \"mkdir -p \\\$DATA_ROOT/{checkpoints,tokenizers,code}\",
                \"echo '=== Downloading checkpoint ==='          \",
                \"aws s3 cp $CKPT_S3 \\\$DATA_ROOT/checkpoints/model.pt --only-show-errors\",
                \"aws s3 cp $TOKENIZER_S3 \\\$DATA_ROOT/tokenizers/tokenizer.model --only-show-errors\",
                \"echo '=== Downloading code bundle ==='         \",
                \"aws s3 cp $CODE_BUNDLE_S3 /tmp/bundle.tar.gz --only-show-errors\",
                \"rm -rf \\\$DATA_ROOT/code/titanProject\",
                \"tar -xzf /tmp/bundle.tar.gz -C \\\$DATA_ROOT/code\",
                \"rm -f /tmp/bundle.tar.gz\",
                \"echo '=== Installing dependencies ==='         \",
                \"python3 -m pip install -q --no-cache-dir numpy==1.26.4\",
                \"python3 -m pip install -q --upgrade --no-cache-dir torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121\",
                \"python3 -m pip install -q --no-cache-dir sentencepiece pyyaml boto3\",
                \"python3 -m pip install -q --no-cache-dir titans-pytorch==0.5.3 --no-deps\",
                \"python3 -m pip install -q --no-cache-dir einops==0.8.2 einx==0.4.2 hyper-connections==0.4.9 axial-positional-embedding==0.3.12 assoc-scan==0.0.4 ema-pytorch==0.7.9 tqdm fire loguru orjson tensordict==0.11.0 x-transformers==2.17.7 rotary-embedding-torch==0.8.9 ninja pyvers cloudpickle frozendict --no-deps\",
                \"echo '=== Writing config ==='                  \",
                \"python3 -c \\\"import yaml; from pathlib import Path; src=Path('\\\$DATA_ROOT/code/model_training/titanProject/configs/config_gpt_medium.yaml'); cfg=yaml.safe_load(src.read_text()); cfg['data']['tokenizer_path']='\\\$DATA_ROOT/tokenizers/tokenizer.model'; Path('\\\$DATA_ROOT/code/model_training/titanProject/configs/config_serve.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))\\\"\",
                \"echo '=== Starting code review server ==='     \",
                \"cd \\\$DATA_ROOT/code/model_training/titanProject\",
                \"nohup python3 serve_codereview.py --config configs/config_serve.yaml --ckpt \\\$DATA_ROOT/checkpoints/model.pt --port $SERVE_PORT --device auto > /var/log/codereview-serve.log 2>&1 &\",
                \"sleep 15\",
                \"curl -sf http://localhost:$SERVE_PORT/health && echo 'Server is UP' || echo 'WARNING: server not yet ready'\"
            ],
            \"executionTimeout\": [\"1800\"]
        }" \
        --query 'Command.CommandId' \
        --output text --no-cli-pager)

    echo "SSM command: $cmd_id"
    echo "Monitoring startup (checkpoint download + model load ~2-3 min)..."

    for i in $(seq 1 24); do
        sleep 15
        job_status=$(aws ssm list-command-invocations \
            --command-id "$cmd_id" \
            --no-cli-pager \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['CommandInvocations'][0]['Status'] if d['CommandInvocations'] else 'Pending')" 2>/dev/null || echo "Pending")
        echo "  [$i] Status: $job_status"
        if [ "$job_status" = "Success" ]; then
            echo ""
            echo "=== Code Review server is ready ==="
            echo "  Health:  curl http://$ip:$SERVE_PORT/health"
            echo "  Review:  curl -X POST http://$ip:$SERVE_PORT/v1/review -H 'Content-Type: application/json' -d '{\"file_path\": \"api/src/foo.ts\", \"diff\": \"+ const x = any\"}'"
            echo "  Batch:   POST http://$ip:$SERVE_PORT/v1/review/batch"
            echo ""
            echo "Set the env var:"
            echo "  export CODEREVIEW_HOST=$ip"
            return
        elif [ "$job_status" = "Failed" ] || [ "$job_status" = "TimedOut" ]; then
            echo "SSM command failed ($job_status). Check logs:"
            echo "  aws ssm get-command-invocation --command-id $cmd_id --instance-id $iid --no-cli-pager"
            return 1
        fi
    done

    echo ""
    echo "Startup still in progress. Check manually:"
    echo "  curl http://$ip:$SERVE_PORT/health"
    echo "  export CODEREVIEW_HOST=$ip"
}

cmd_status() {
    echo "=== Code Review instance status ==="
    existing="$(find_instance)"
    if [ -z "$existing" ]; then
        echo "No wintermute-codereview-serve instance found."
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
        echo "  export CODEREVIEW_HOST=$ip"
        echo ""
        echo "Checking server health..."
        curl -sf --connect-timeout 5 "http://$ip:$SERVE_PORT/health" 2>/dev/null \
            && echo "" \
            || echo "  Server not responding (may still be starting up)"
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
    echo "Instance terminating. Remember to unset CODEREVIEW_HOST."
}

case "${1:-help}" in
    launch)    cmd_launch ;;
    status)    cmd_status ;;
    stop)      cmd_stop ;;
    terminate) cmd_terminate ;;
    *)
        echo "Usage: $0 {launch|status|stop|terminate}"
        echo ""
        echo "  launch     Launch EC2 + deploy code review server"
        echo "  status     Show instance state, IP, and server health"
        echo "  stop       Stop instance (preserves EBS, saves cost)"
        echo "  terminate  Terminate instance (destroys everything)"
        ;;
esac
