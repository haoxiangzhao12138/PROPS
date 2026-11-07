#!/bin/bash
# Serve the two scoring models of PROPS with vLLM, one GPU per server.
#   judge     google/gemma-4-31B-it   ports 9000+i   (Follow and Exec questions)
#   grounder  Qwen/Qwen3.8-27B        ports 8000+i   (boxes the prop on the drawn evidence frames of Reuse pairs)
# Usage:
#   bash scripts/serve_judge.sh judge 0-3        # judge on GPUs 0..3
#   bash scripts/serve_judge.sh grounder 4-7     # grounder on GPUs 4..7
# then pass --judge http://HOST:9000/v1,http://HOST:9001/v1,... and --grounder http://HOST:8004/v1,... to props score.
# One H100/H200 class GPU (80 GB or more) holds one server. Logs go to logs/vllm/.
set -e
ROLE=${1:?judge or grounder}
RANGE=${2:-0-0}
FIRST=${RANGE%-*}; LAST=${RANGE#*-}
case $ROLE in
  judge)    M=${JUDGE_MODEL:-google/gemma-4-31B-it}; BASE=9000; LEN=16384; MM='{"image": 8}' ;;
  grounder) M=${GROUNDER_MODEL:-Qwen/Qwen3.8-27B};   BASE=8000; LEN=32768; MM='{"image": 12, "video": 0}' ;;
  *) echo "role must be judge or grounder"; exit 1 ;;
esac
export VLLM_NO_USAGE_STATS=1
mkdir -p logs/vllm
for i in $(seq "$FIRST" "$LAST"); do
  CUDA_VISIBLE_DEVICES=$i nohup vllm serve "$M" --served-model-name "$ROLE" \
    --host 0.0.0.0 --port $((BASE + i)) --max-model-len $LEN --gpu-memory-utilization ${MEM:-0.90} \
    --max-num-seqs 128 --limit-mm-per-prompt "$MM" --enable-prefix-caching \
    > "logs/vllm/$ROLE-$i.log" 2>&1 < /dev/null &
  echo "$ROLE on GPU $i: http://$(hostname):$((BASE + i))/v1"
done
