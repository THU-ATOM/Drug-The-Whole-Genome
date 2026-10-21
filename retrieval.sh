#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root; paths below are relative to it.
: "${POCKET_PATH:?Set POCKET_PATH to a pocket.lmdb file; see docs/QUICKSTART.md}"
MOL_PATH="${MOL_PATH:-mols.lmdb}"
FOLD_VERSION="${FOLD_VERSION:-6_folds}"
USE_CACHE="${USE_CACHE:-true}"
SAVE_PATH="${SAVE_PATH:-results/retrieval.csv}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Preserve the previous wrapper's CPU default; GPU is explicit.
device_args=()
case "${DEVICE:-cpu}" in
    cpu) device_args+=(--cpu) ;;
    cuda) device_args+=(--fp16 --fp16-init-scale 4 --fp16-scale-window 256) ;;
    *) echo "DEVICE must be cpu or cuda" >&2; exit 2 ;;
esac

"${PYTHON:-python}" ./unimol/retrieval.py ./dict --user-dir ./unimol --valid-subset test \
    --num-workers 8 --ddp-backend=c10d --batch-size 4 \
    --task drugclip --loss in_batch_softmax --arch drugclip --max-pocket-atoms 511 \
    "${device_args[@]}" --seed 1 --log-interval 100 --log-format simple \
    --mol-path "$MOL_PATH" --pocket-path "$POCKET_PATH" \
    --fold-version "$FOLD_VERSION" --use-cache "$USE_CACHE" --save-path "$SAVE_PATH"
