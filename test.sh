#!/usr/bin/env bash
set -euo pipefail
TASK="${TASK:-PCBA}"
USE_FOLDS="${USE_FOLDS:-false}"
WEIGHT_PATH="${WEIGHT_PATH:-./data/benchmark_weights/90.pt}"
RESULTS_PATH="${RESULTS_PATH:-./test}"
BATCH_SIZE="${BATCH_SIZE:-64}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
case "$TASK" in DUDE|PCBA) ;; *) echo "TASK must be DUDE or PCBA" >&2; exit 2 ;; esac

"${PYTHON:-python}" ./unimol/test.py ./dict --user-dir ./unimol --valid-subset test \
    --results-path "$RESULTS_PATH" --num-workers 8 --ddp-backend=c10d --batch-size "$BATCH_SIZE" \
    --task drugclip --loss in_batch_softmax --arch drugclip \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
    --use-folds "$USE_FOLDS" --path "$WEIGHT_PATH" \
    --log-interval 100 --log-format simple --max-pocket-atoms 511 --test-task "$TASK"
