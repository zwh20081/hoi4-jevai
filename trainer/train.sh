#!/usr/bin/env bash
# Train in a detached tmux session named after the run; resumes temp/train/<name>/state.pt when present.
#   trainer/train.sh hoi4-v1 [extra trainer args]       watch: tmux attach -t hoi4-v1   (detach: Ctrl+B D)
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:-hoi4-v1}
shift || true
if [ -f "temp/train/$NAME/state.pt" ]; then
  ARGS="--out temp/train/$NAME --resume $*"
else
  ARGS="--out temp/train/$NAME --bf16 --bs 16 --accum 2 --epochs 1 --eval-every 1000 --save-every 500 --val-states 1500 $*"
fi
PY=${PY:-python}
tmux new-session -d -s "$NAME" "PYTHONWARNINGS=ignore $PY -m trainer.train $ARGS 2>&1 | tee -a temp/train/$NAME.console.log; echo; echo 'training exited'; exec bash"
echo "started tmux session $NAME: $PY -m trainer.train $ARGS"
