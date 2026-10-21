#!/usr/bin/env bash
set -euo pipefail
if (( $# != 3 )); then
    echo "Usage: bash encode_mols.sh GPU_ID MOL_PATH SAVE_DIR" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$1"
range_args=(--start "${START:-0}")
if [[ -n "${END:-}" ]]; then range_args+=(--end "$END"); fi
device_args=()
case "${DEVICE:-cpu}" in
    cpu) device_args+=(--cpu) ;;
    cuda) ;;
    *) echo "DEVICE must be cpu or cuda" >&2; exit 2 ;;
esac
"${PYTHON:-python}" ./unimol/encode_mols.py ./dict --user-dir ./unimol --valid-subset test \
    "${device_args[@]}" --num-workers 0 --ddp-backend=c10d --batch-size "${BATCH_SIZE:-256}" \
    --task drugclip --loss in_batch_softmax --arch drugclip --max-pocket-atoms 256 \
    --seed 1 --log-interval 100 --log-format simple \
    --mol-path "$2" --save-dir "$3" "${range_args[@]}" --write-h5
