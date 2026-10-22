#!/usr/bin/env bash
# Capture argv instead of importing the model or requiring a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
capture=$(mktemp ./tests/.capture.XXXXXX)
fake_python=$(mktemp ./tests/.python.XXXXXX)
trap 'rm -f -- "$capture" "$fake_python"' EXIT
cat > "$fake_python" <<'STUB'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$CAPTURE"
STUB
chmod +x "$fake_python"
export CAPTURE="$capture" PYTHON="$fake_python"
unset END START DEVICE USE_FOLDS TASK TARGETS

expect_pair() {
    local previous='' current
    while IFS= read -r current; do
        if [[ "$previous" == "$1" && "$current" == "$2" ]]; then return; fi
        previous="$current"
    done < "$capture"
    echo "Missing argv pair: $1 $2" >&2
    exit 1
}
expect_absent() {
    local current
    while IFS= read -r current; do
        if [[ "$current" == "$1" ]]; then echo "Unexpected argument $1" >&2; exit 1; fi
    done < "$capture"
}

POCKET_PATH='data/pocket with spaces.lmdb' MOL_PATH='data/my mols.lmdb' \
SAVE_PATH='results/my scores.csv' USE_CACHE=false DEVICE=cuda bash retrieval.sh
expect_pair --pocket-path 'data/pocket with spaces.lmdb'
expect_pair --mol-path 'data/my mols.lmdb'
expect_pair --save-path 'results/my scores.csv'
expect_pair --use-cache false
expect_absent --cpu
POCKET_PATH='pocket.lmdb' DEVICE=cpu bash retrieval.sh
expect_absent --fp16

TARGETS='target_a target_b' RESULTS_PATH='results/my run' TASK=DUDE USE_FOLDS=true WEIGHT_PATH='weights/my model.pt' bash test.sh
expect_pair --test-task DUDE
expect_pair --use-folds true
expect_pair --targets target_a
expect_pair target_a target_b
expect_pair --results-path 'results/my run'
expect_pair --path 'weights/my model.pt'
bash test.sh
expect_pair --use-folds false

bash encode_mols.sh 0 'data/my molecules' 'results/my embeddings'
expect_pair --mol-path 'data/my molecules'
expect_pair --save-dir 'results/my embeddings'
expect_absent --end
START=5 END=10 DEVICE=cuda bash encode_mols.sh 0 data/mols results/embeddings
expect_pair --start 5
expect_pair --end 10
expect_absent --cpu
bash encode_pocket.sh 0 'data/my pockets'
expect_pair --pocket-dir 'data/my pockets'

if (unset POCKET_PATH; bash retrieval.sh) >/dev/null 2>&1; then exit 1; fi
if DEVICE=typo POCKET_PATH=pocket.lmdb bash retrieval.sh >/dev/null 2>&1; then exit 1; fi
if TASK=typo bash test.sh >/dev/null 2>&1; then exit 1; fi
if bash encode_mols.sh >/dev/null 2>&1; then exit 1; fi
if bash encode_pocket.sh >/dev/null 2>&1; then exit 1; fi
echo 'Shell wrapper checks passed.'
