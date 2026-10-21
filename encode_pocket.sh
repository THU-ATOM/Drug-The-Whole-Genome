#!/usr/bin/env bash
set -euo pipefail
if (( $# != 2 )); then
    echo "Usage: bash encode_pocket.sh GPU_ID POCKET_DIR" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$1"
"${PYTHON:-python}" ./unimol/encode_pockets.py ./dict --user-dir ./unimol --valid-subset test \
    --num-workers 1 --ddp-backend=c10d --batch-size "${BATCH_SIZE:-8}" \
    --task drugclip --loss in_batch_softmax --arch drugclip --max-pocket-atoms 511 \
    --seed 1 --log-interval 100 --log-format simple --pocket-dir "$2"
