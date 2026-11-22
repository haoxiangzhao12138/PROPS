#!/bin/bash
# One-time setup of PROPS on a Linux machine. Run from the repository root:
#
#   bash scripts/setup.sh                 # props in .venv + PROPS-Benchmark in data/
#   bash scripts/setup.sh --judge         # also vLLM in .venv-vllm and the judge and grounder weights in models/
#   bash scripts/setup.sh --judge --train # also PROPS-Train (about 24 GB)
#
# Options:
#   --judge        install vLLM in its own environment (.venv-vllm) and download the two scoring models (118 GB)
#   --train        download PROPS-Train as well
#   --no-data      skip the dataset download
#   --ego          also install the extras of `props rebuild-ego` (PyAV, OpenCV)
#   --python BIN   Python to build the environments with (default python3; 3.9+, and 3.10 to 3.14 with --judge,
#                  which vLLM needs)
# A step that is already done is skipped, so the script can be run again after an interruption.
set -e
cd "$(dirname "$0")/.."
. scripts/versions.sh
JUDGE=0; TRAIN=0; DATA=1; EGO=0; PY=python3
while [ $# -gt 0 ]; do
  case $1 in
    --judge) JUDGE=1 ;;
    --train) TRAIN=1 ;;
    --no-data) DATA=0 ;;
    --ego) EGO=1 ;;
    --python) PY=$2; shift ;;
    -h|--help) sed -n 2,16p "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 2 ;;
  esac
  shift
done
step() { echo; echo "== $*"; }
# Containers such as the NGC PyTorch images pin their own torch through PIP_CONSTRAINT and put it first on
# LD_LIBRARY_PATH. Both are meant for the image's Python and break the separate environments made here.
unset PIP_CONSTRAINT
LD=; IFS=: read -r -a P <<< "${LD_LIBRARY_PATH:-}"
for p in "${P[@]}"; do case $p in */dist-packages/*|*/site-packages/*|"") ;; *) LD=${LD:+$LD:}$p ;; esac; done
export LD_LIBRARY_PATH=$LD
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 9))' || { echo "$PY is older than Python 3.9"; exit 2; }
if [ $JUDGE = 1 ] && ! "$PY" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] < (3, 15))'; then
  echo "--judge installs vLLM $VLLM_VERSION, which needs Python 3.10 to 3.14; pass one with --python"; exit 2
fi

step "1. props in .venv"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
NEED=import\ props.cli; [ $EGO = 1 ] && NEED="$NEED, av, cv2"
if .venv/bin/python -c "$NEED" 2> /dev/null && [ "$(.venv/bin/python -c 'import props, os; print(os.path.dirname(os.path.dirname(props.__file__)))')" = "$PWD" ]; then
  echo "ok: .venv/bin/props (already installed)"
else
  .venv/bin/pip install -q --upgrade pip
  if [ $EGO = 1 ]; then .venv/bin/pip install -q -e ".[ego]"; else .venv/bin/pip install -q -e .; fi
  .venv/bin/props -h > /dev/null && echo "ok: .venv/bin/props"
fi

if [ $DATA = 1 ]; then
  step "2. datasets in data/"
  .venv/bin/props download --out data --revision "$BENCH_REV"
  [ $TRAIN = 1 ] && .venv/bin/props download --out data --train-only --revision "$TRAIN_REV"
fi

if [ $JUDGE = 1 ]; then
  step "3. vLLM $VLLM_VERSION in .venv-vllm (kept apart from props and from your generator)"
  [ -x .venv-vllm/bin/python ] || "$PY" -m venv .venv-vllm
  if [ "$(.venv-vllm/bin/python -c 'import vllm; print(vllm.__version__)' 2> /dev/null)" = "$VLLM_VERSION" ]; then
    echo "ok: .venv-vllm/bin/vllm $VLLM_VERSION (already installed)"
  else
    .venv-vllm/bin/pip install -q --upgrade pip
    .venv-vllm/bin/pip install -q "vllm==$VLLM_VERSION"
    .venv-vllm/bin/vllm --help > /dev/null && echo "ok: .venv-vllm/bin/vllm"
  fi

  step "4. scoring models in models/"
  for spec in "$JUDGE_REPO@$JUDGE_REV" "$GROUNDER_REPO@$GROUNDER_REV"; do
    repo=${spec%@*}; rev=${spec#*@}; dst=models/${repo#*/}
    if [ -f "$dst/.props_rev" ] && [ "$(cat "$dst/.props_rev")" = "$rev" ]; then echo "ok: $dst ($rev)"; continue; fi
    echo "$repo @ $rev -> $dst"
    .venv/bin/python - "$repo" "$rev" "$dst" <<'EOF'
import sys
from huggingface_hub import snapshot_download
from props.data import fetch
repo, rev, dst = sys.argv[1:]
fetch(lambda: snapshot_download(repo_id=repo, revision=rev, local_dir=dst, max_workers=8))
EOF
    echo "$rev" > "$dst/.props_rev"
  done
fi

step "done"
echo "Next: draw the frames of your model (README, Evaluate your model), then"
echo "  bash scripts/evaluate.sh runs/<name> --gpus 0,1"
