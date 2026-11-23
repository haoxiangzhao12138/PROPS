#!/bin/bash
# Score a run folder from start to end: start the judge and the grounder with vLLM, wait until they answer, score
# every pair, print the board, write <run>/props_board.json and stop the servers.
#
#   bash scripts/evaluate.sh runs/mymodel                         # GPUs 0,1: judge on 0, grounder on 1
#   bash scripts/evaluate.sh runs/mymodel --gpus 0-7              # four servers of each
#   bash scripts/evaluate.sh runs/mymodel --gpus 0-3 --tp 2       # one server of each over two GPUs (48 GB cards)
#   bash scripts/evaluate.sh runs/mymodel --table frames.csv      # first copy frames listed in a CSV (item,step,path)
#   bash scripts/evaluate.sh runs/mymodel --judge URL --grounder URL   # servers already running elsewhere
#
# Options:
#   --split S        full (default), phase_a or open
#   --bench DIR      PROPS-Benchmark folder (default data/PROPS-Benchmark)
#   --gpus LIST      GPUs for the servers, the first half for the judge and the second half for the grounder
#   --tp N           GPUs per server (default 1, which needs 80 GB per GPU)
#   --table CSV      run `props collect` with this table first
#   --allow-missing  score even if frames are missing (each missing frame fails its side)
#   --judge, --grounder  base URLs of running servers (give both); no server is started
#   --judge-model, --grounder-model  served names of those servers (default judge, grounder)
# Environment: WAIT (seconds to wait for the servers to load, default 2400), and those of serve_judge.sh.
set -e
cd "$(dirname "$0")/.."
RUN=${1:?usage: evaluate.sh RUN_DIR [options], see the top of this file}; shift
SPLIT=full; BENCH=data/PROPS-Benchmark; GPUS=0,1; TP=1; TABLE=; ALLOW=; JURL=; GURL=; JM=judge; GM=grounder
while [ $# -gt 0 ]; do
  case $1 in
    --split) SPLIT=$2; shift ;;
    --bench) BENCH=$2; shift ;;
    --gpus) GPUS=$2; shift ;;
    --tp) TP=$2; shift ;;
    --table) TABLE=$2; shift ;;
    --allow-missing) ALLOW=--allow-missing ;;
    --judge) JURL=$2; shift ;;
    --grounder) GURL=$2; shift ;;
    --judge-model) JM=$2; shift ;;
    --grounder-model) GM=$2; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
if [ -n "$JURL$GURL" ] && { [ -z "$JURL" ] || [ -z "$GURL" ]; }; then
  echo "give both --judge and --grounder, or neither to start both servers here"; exit 2
fi
PROPS=props; [ -x .venv/bin/props ] && PROPS=.venv/bin/props
COMMON=(--bench "$BENCH" --split "$SPLIT")

if [ -n "$TABLE" ]; then
  echo "== collect frames from $TABLE"
  $PROPS collect "${COMMON[@]}" --out "$RUN" --table "$TABLE" || { echo "fix the rows above, then run again"; exit 1; }
fi

echo "== check $RUN"
if ! $PROPS check "${COMMON[@]}" --outputs "$RUN" && [ -z "$ALLOW" ]; then
  echo "frames are missing; draw them (props export lists them) or pass --allow-missing"; exit 2
fi

if [ -z "$JURL" ]; then
  case $GPUS in *-*) GPUS=$(seq -s, "${GPUS%-*}" "${GPUS#*-}") ;; esac
  IFS=, read -r -a G <<< "$GPUS"
  HALF=$(( ${#G[@]} / 2 ))
  if [ $HALF -lt "$TP" ] || [ $(( HALF % TP )) -ne 0 ]; then
    echo "--gpus needs an even number of GPUs, at least 2*TP, and half of them a multiple of TP"; exit 2
  fi
  JG=$(IFS=,; echo "${G[*]:0:HALF}"); GG=$(IFS=,; echo "${G[*]:HALF}")
  stop() {  # SIGTERM to every server, and SIGKILL to those still alive after 30 s (a server stuck in start-up)
    local ps=() p
    for r in judge grounder; do
      [ -f "logs/vllm/$r.pids" ] && while read -r p; do ps+=("$p"); done < "logs/vllm/$r.pids"
    done
    for p in "${ps[@]}"; do kill "$p" 2> /dev/null || true; done
    for _ in $(seq 30); do
      for p in "${ps[@]}"; do kill -0 "$p" 2> /dev/null && break; p=; done
      [ -z "$p" ] && return 0
      sleep 1
    done
    for p in "${ps[@]}"; do pkill -KILL -P "$p" 2> /dev/null; kill -KILL "$p" 2> /dev/null || true; done
  }
  trap stop EXIT
  echo "== start the judge on GPU $JG and the grounder on GPU $GG"
  TP=$TP bash scripts/serve_judge.sh judge "$JG"
  TP=$TP bash scripts/serve_judge.sh grounder "$GG"
  JURL=$(paste -sd, logs/vllm/judge.urls); GURL=$(paste -sd, logs/vllm/grounder.urls)
  echo "== wait for the servers (loading the weights takes a few minutes)"
  T0=$(date +%s)
  for u in $(tr ',' ' ' <<< "$JURL,$GURL"); do
    until curl -sf "$u/models" > /dev/null; do
      for r in judge grounder; do
        while read -r p; do
          kill -0 "$p" 2> /dev/null || { echo "a $r server exited; the first errors in its log:"
            grep -h -m 3 -E "Error|error:" logs/vllm/$r-*.log | grep -v -E "KeyboardInterrupt|terminated" | head -n 5
            echo "(full logs in logs/vllm/)"; exit 1; }
        done < "logs/vllm/$r.pids"
      done
      [ $(( $(date +%s) - T0 )) -lt "${WAIT:-2400}" ] || { echo "servers not ready after ${WAIT:-2400} s, see logs/vllm/"; exit 1; }
      sleep 10
    done
    echo "ready: $u"
  done
fi

echo "== score and board"
$PROPS evaluate "${COMMON[@]}" --outputs "$RUN" --judge "$JURL" --grounder "$GURL" \
  --judge-model "$JM" --grounder-model "$GM" $ALLOW
