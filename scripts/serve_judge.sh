#!/bin/bash
# Serve the two scoring models of PROPS with vLLM.
#   judge     google/gemma-4-31B-it   ports 9000+g   (Follow and Exec questions)
#   grounder  Qwen/Qwen3.8-27B        ports 8000+g   (boxes the prop on the drawn evidence frames of Reuse pairs)
# g is the first GPU of the server. Usage:
#   bash scripts/serve_judge.sh judge 0,1,2,3          # four judge servers, one GPU each (80 GB or more per GPU)
#   bash scripts/serve_judge.sh grounder 4,5,6,7
#   TP=2 bash scripts/serve_judge.sh judge 0,1,2,3     # two judge servers over two GPUs each (smaller GPUs)
# then pass --judge http://HOST:9000/v1,http://HOST:9001/v1,... and --grounder http://HOST:8004/v1,... to props.
# Ranges like 0-3 work too. Environment:
#   JUDGE_MODEL, GROUNDER_MODEL  weights (default models/<name> if setup.sh downloaded them, else the pinned HF repo)
#   HOST (address to listen on, default 127.0.0.1; 0.0.0.0 serves other machines), ADVERTISE (host name or IP that
#   other machines use in the URLs, default HOST, or this machine's name when HOST is 0.0.0.0),
#   TP (GPUs per server, default 1), MEM (default 0.90)
#   VLLM (vllm binary, default .venv-vllm/bin/vllm if it exists, else vllm in PATH)
#   CUDA_COMPAT (folder of NVIDIA's CUDA 13 forward-compatibility libraries, for drivers older than release 580)
# The URLs of the servers go to logs/vllm/<role>.urls, their pids to logs/vllm/<role>.pids and their logs to
# logs/vllm/<role>-<g>.log. scripts/evaluate.sh starts and stops the servers itself.
set -e
ROLE=${1:?judge or grounder}
GPUS=${2:-0}
cd "$(dirname "$0")/.."
. scripts/versions.sh
case $ROLE in
  judge)    M=$JUDGE_MODEL;    HF=$JUDGE_REPO;    REV=$JUDGE_REV;    BASE=9000; LEN=16384; MM='{"image": 8}' ;;
  grounder) M=$GROUNDER_MODEL; HF=$GROUNDER_REPO; REV=$GROUNDER_REV; BASE=8000; LEN=32768; MM='{"image": 12, "video": 0}' ;;
  *) echo "role must be judge or grounder"; exit 2 ;;
esac
EXTRA=()
if [ -z "$M" ]; then
  M=models/${HF#*/}
  [ -d "$M" ] || { M=$HF; EXTRA=(--revision "$REV"); }  # not downloaded by setup.sh: vLLM fetches the pinned revision
fi
NAMES=("$ROLE")  # a second served name <role>@<revision> tells clients which revision of the weights is served
if [ "$M" = "$HF" ]; then NAMES+=("$ROLE@$REV"); elif [ -f "$M/.props_rev" ]; then NAMES+=("$ROLE@$(cat "$M/.props_rev")"); fi
if [ -z "$VLLM" ]; then VLLM=vllm; [ -x .venv-vllm/bin/vllm ] && VLLM=.venv-vllm/bin/vllm; fi
case $GPUS in *-*) GPUS=$(seq -s, "${GPUS%-*}" "${GPUS#*-}") ;; esac
IFS=, read -r -a G <<< "$GPUS"
TP=${TP:-1}
HOST=${HOST:-127.0.0.1}
if [ -z "$ADVERTISE" ]; then ADVERTISE=$HOST; [ "$HOST" = 0.0.0.0 ] && ADVERTISE=$(hostname -f 2> /dev/null || hostname); fi
if [ $(( ${#G[@]} % TP )) -ne 0 ]; then echo "the number of GPUs (${#G[@]}) is not a multiple of TP=$TP"; exit 2; fi
export VLLM_NO_USAGE_STATS=1
mkdir -p logs/vllm
# drop library paths of another Python's packages (the NGC images put their torch first), so that vLLM loads the
# torch of its own environment
LD=; IFS=: read -r -a P <<< "${LD_LIBRARY_PATH:-}"
for p in "${P[@]}"; do case $p in */dist-packages/*|*/site-packages/*|"") ;; *) LD=${LD:+$LD:}$p ;; esac; done
export LD_LIBRARY_PATH=$LD
# vLLM 0.30.0 from PyPI is built for CUDA 13, which needs an NVIDIA driver of release 580 or newer. On an older
# driver of a data-center GPU, point CUDA_COMPAT at the folder of NVIDIA's CUDA 13 forward-compatibility libraries
# (package cuda-compat-13-0, installed to /usr/local/cuda-13.0/compat).
[ -n "$CUDA_COMPAT" ] && export LD_LIBRARY_PATH=$CUDA_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
VPY=$(dirname "$(command -v "$VLLM")")/python
if [ -x "$VPY" ] && ! CUDA_VISIBLE_DEVICES=${GPUS%%,*} "$VPY" -c 'import torch; torch.zeros(1).cuda()' 2> logs/vllm/cuda_check.log; then
  echo "the torch of $VPY cannot use the GPU (logs/vllm/cuda_check.log):"; tail -n 1 logs/vllm/cuda_check.log
  echo "vLLM $VLLM_VERSION needs CUDA 13, that is an NVIDIA driver of release 580 or newer; with an older driver set"
  echo "CUDA_COMPAT to NVIDIA's CUDA 13 forward-compatibility libraries (cuda-compat-13-0), see docs/scoring.md"
  exit 2
fi
for p in $(cat "logs/vllm/$ROLE.pids" 2> /dev/null); do  # servers of an earlier call still running
  if kill -0 "$p" 2> /dev/null; then echo "a $ROLE server of an earlier call still runs (pid $p); stop it first"; exit 2; fi
done
: > "logs/vllm/$ROLE.pids"; : > "logs/vllm/$ROLE.urls"
for ((k = 0; k < ${#G[@]}; k += TP)); do
  DEV=$(IFS=,; echo "${G[*]:k:TP}")
  PORT=$((BASE + G[k]))
  CUDA_VISIBLE_DEVICES=$DEV nohup "$VLLM" serve "$M" "${EXTRA[@]}" --served-model-name "${NAMES[@]}" \
    --host "$HOST" --port $PORT --tensor-parallel-size "$TP" --max-model-len $LEN \
    --gpu-memory-utilization "${MEM:-0.90}" --max-num-seqs 128 --limit-mm-per-prompt "$MM" --enable-prefix-caching \
    > "logs/vllm/$ROLE-${G[k]}.log" 2>&1 < /dev/null &
  echo $! >> "logs/vllm/$ROLE.pids"
  echo "http://$ADVERTISE:$PORT/v1" >> "logs/vllm/$ROLE.urls"
  echo "$ROLE on GPU $DEV: http://$ADVERTISE:$PORT/v1   (log logs/vllm/$ROLE-${G[k]}.log)"
done
