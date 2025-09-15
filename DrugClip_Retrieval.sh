#!/usr/bin/env bash
PYTHONBIN="${PYTHONBIN:-$(which python)}"
# Ensure we're running from the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"
# DrugClip Retrieval (CLI)
# Required:
#   -m  Path to ligand .lmdb
#   -n  Custom cache name (string)
#   -p  Path to preprocessed pocket .lmdb
#   -S  Save/output directory
#   -D  Custom cache BASE directory (cache stored at <base>/<cache_name>)
#
# Optional:
#   -u  USE_CACHE (True|False)                 [default: False]
#   -t  TOPK_PERCENT (float)                   [default: 100.0]
#   -f  FOLD_VERSION                           [default: 6_folds]
#   -x  MULTI_CONF (True|False)                [default: False]
#   -g  CUDA_VISIBLE_DEVICES (e.g., 0 or 0,1)  [default: 0]
#   -l  Explicit log file path                 [default: <SAVE_PATH>/retrieval_<cache>_<timestamp>.log]

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash DrugClip_Retrieval.sh \
    -m <ligands.lmdb> \
    -n <cache_name> \
    -p <pocket.lmdb> \
    -S <save_dir> \
    -D <cache_base_dir> \
    [-u True|False] \
    [-t <topk_percent>] \
    [-f <fold_version>] \
    [-x True|False] \
    [-g <cuda_visible_devices>] \
    [-l <log_file>]
USAGE
}

# -------- Defaults --------
USE_CACHE="False"
TOPK_PERCENT="100.0"
FOLD_VERSION="6_folds"
MULTI_CONF="False"
CUDA_DEVICES="0"
LOG_FILE=""

# -------- Parse args --------
while getopts ":m:n:p:S:D:u:t:f:x:g:l:h" opt; do
  case "$opt" in
    m) MOL_PATH="$OPTARG" ;;
    n) CACHE_NAME="$OPTARG" ;;
    p) POCKET_PATH="$OPTARG" ;;
    S) SAVE_PATH="$OPTARG" ;;
    D) CACHE_BASE_DIR="$OPTARG" ;;
    u) USE_CACHE="$OPTARG" ;;
    t) TOPK_PERCENT="$OPTARG" ;;
    f) FOLD_VERSION="$OPTARG" ;;
    x) MULTI_CONF="$OPTARG" ;;
    g) CUDA_DEVICES="$OPTARG" ;;
    l) LOG_FILE="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Error: Invalid option -$OPTARG"; usage; exit 1 ;;
    :)  echo "Error: Option -$OPTARG requires an argument"; usage; exit 1 ;;
  esac
done

# -------- Validate required --------
: "${MOL_PATH:?Missing -m <ligands.lmdb>}"
: "${CACHE_NAME:?Missing -n <cache_name>}"
: "${POCKET_PATH:?Missing -p <pocket.lmdb>}"
: "${SAVE_PATH:?Missing -S <save_dir>}"
: "${CACHE_BASE_DIR:?Missing -D <cache_base_dir>}"

# -------- Prep paths --------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CUSTOM_CACHE_DIR="${CACHE_BASE_DIR%/}/${CACHE_NAME}"
CUSTOM_SAVE_DIR="${SAVE_PATH%/}/${CACHE_NAME}"   # <-- NEW
mkdir -p "$CUSTOM_CACHE_DIR"
mkdir -p "$CUSTOM_SAVE_DIR"


# -------- Log file default --------
if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="${CUSTOM_SAVE_DIR}/retrieval_${CACHE_NAME}_${TIMESTAMP}.log"  # <-- UPDATED
fi

# -------- Start timer --------
start=$(date +%s)

# -------- Export env for cache override --------
export CUSTOM_CACHE_DIR="$CUSTOM_CACHE_DIR"

# -------- Echo config --------
echo "=== DrugClip Retrieval Config ===" | tee "$LOG_FILE"
echo "Ligand LMDB         : $MOL_PATH"       | tee -a "$LOG_FILE"
echo "Pocket LMDB         : $POCKET_PATH"    | tee -a "$LOG_FILE"
echo "Cache Base Dir      : $CACHE_BASE_DIR" | tee -a "$LOG_FILE"
echo "Custom Cache Dir    : $CUSTOM_CACHE_DIR" | tee -a "$LOG_FILE"
echo "Cache Name          : $CACHE_NAME"     | tee -a "$LOG_FILE"
echo "Save Path           : $SAVE_PATH"      | tee -a "$LOG_FILE"
echo "Use Cache           : $USE_CACHE"      | tee -a "$LOG_FILE"
echo "TopK Percent        : $TOPK_PERCENT"   | tee -a "$LOG_FILE"
echo "Fold Version        : $FOLD_VERSION"   | tee -a "$LOG_FILE"
echo "Multi Conf          : $MULTI_CONF"     | tee -a "$LOG_FILE"
echo "CUDA Devices        : $CUDA_DEVICES"   | tee -a "$LOG_FILE"
echo "Log File            : $LOG_FILE"       | tee -a "$LOG_FILE"
echo "Timestamp           : $TIMESTAMP"      | tee -a "$LOG_FILE"

# -------- Run retrieval --------
# Note: Keep --cpu to match your working command behavior.
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHONBIN" ./unimol/retrieval.py --user-dir ./unimol ${data_path:-} "./dict" --valid-subset test \
  --num-workers 8 --ddp-backend=c10d --batch-size 4 \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --max-pocket-atoms 511 \
  --cpu \
  --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 \
  --log-interval 100 --log-format simple \
  --mol-path "$MOL_PATH" \
  --pocket-path "$POCKET_PATH" \
  --fold-version "$FOLD_VERSION" \
  --use-cache "$USE_CACHE" \
  --custom-cache-dir "$CUSTOM_CACHE_DIR" \
  --topk-percent "$TOPK_PERCENT" \
  --save-path "$CUSTOM_SAVE_DIR" \
  --multi-conf "$MULTI_CONF" | tee -a "$LOG_FILE"

# -------- Safety sweep: move any stray retrieval-*.txt from parent into subdir --------
shopt -s nullglob
for f in "${SAVE_PATH%/}"/retrieval-*.txt; do
  mv -f "$f" "$CUSTOM_SAVE_DIR"/
done
shopt -u nullglob

# -------- End timer --------
end=$(date +%s)
runtime=$((end - start))
echo "✅ Retrieval completed in ${runtime} seconds." | tee -a "$LOG_FILE"
