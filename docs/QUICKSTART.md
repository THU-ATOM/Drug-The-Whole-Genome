# Quick start

Complete the [environment checks](INSTALL.md) first. All paths below are relative
to the repository root unless absolute. Replace paths to match the downloaded
archive; do not assume every target archive has the same top-level folder.

## Download inputs

Download the required archives from
[DrugCLIP_data](https://huggingface.co/datasets/bgao95/DrugCLIP_data):
`model_weights.zip`, `targets.zip`, and, for the distributed molecule library,
`encoded_mol_embs.zip`. Extract under `data/` and check the actual layout:

```text
dict/dict_mol.txt
dict/dict_pkt.txt
data/model_weights/6_folds/fold_0.pt ... fold_5.pt
data/encoded_mol_embs/6_folds/fold0.pkl ... fold5.pkl
data/<your target directory>/pocket.lmdb
```

The checkpoint filenames contain an underscore (`fold_0.pt`); cache filenames do
not (`fold0.pkl`). Eight-fold screening requires the corresponding eight
checkpoints and eight caches. `6_folds_filtered` reuses six-fold checkpoints with
the `6_folds_filtered` cache directory. Do not mix cache versions or molecule order.

## Screen the distributed library

Set the actual pocket location, then run:

```bash
POCKET_PATH='./data/targets/YOUR_TARGET/pocket.lmdb' \
USE_CACHE=true SAVE_PATH='./results/screen.csv' \
DEVICE=cuda CUDA_VISIBLE_DEVICES=0 bash retrieval.sh
```

For CPU retrieval replace `DEVICE=cuda` with `DEVICE=cpu`. Missing inputs are
reported before model allocation. The result parent directory is created.
`USE_CACHE=true` uses the cached library; changing `MOL_PATH` does not change it.
The original project instructions specify eight-fold screening for 5HT2A; select
`FOLD_VERSION=8_folds` and download matching inputs for that configuration.

Output is a headerless UTF-8 text file containing `smiles,score`, sorted by
descending score. The existing policy retains the top 2% (rounded down); this
branch additionally retains one result for a nonempty library of fewer than 50
molecules. Scores are retrieval scores, not binding probabilities or affinity units.

## Screen a custom molecule library

Prepare a nonempty single-file molecule LMDB and a pocket LMDB following
[data formats](DATA_FORMAT.md). Then:

```bash
POCKET_PATH='./data/custom/pocket.lmdb' MOL_PATH='./data/custom/mols.lmdb' \
USE_CACHE=false SAVE_PATH='./results/custom.csv' \
DEVICE=cuda CUDA_VISIBLE_DEVICES=0 bash retrieval.sh
```

This path encodes the supplied molecules for each fold. It does not write into
the distributed cache directory. This changes the previous behavior, which could
replace official caches with embeddings from a custom library.

For a small end-to-end check, use a trusted small molecule LMDB and one validated
pocket with the six weights. Verify that output is nonempty and every returned
SMILES belongs to the input. No ready-to-run scientific fixture or expected
checkpoint-level scores are bundled by this refactor.

## Encode molecules separately

```bash
DEVICE=cuda bash encode_mols.sh 0 ./data/custom/mols.lmdb ./results/encoded
```

The wrapper writes HDF5 and processes the full input by default. For the directory
LMDB v2 format only, `START=0 END=1000` selects an index slice of the sorted unique
`success` keys. The single-file loader ignores slicing and logs that fact. The
previous implicit `--end 30000` has been removed. For NumPy output use the Python
entry point with `--write-npy` (and optionally `--write-h5`). See its `--help`.

Do not reuse an HDF5 output path for a different input, order or checkpoint set:
the encoder's existing resume flags do not verify that provenance.

## Encode pockets separately

```bash
bash encode_pocket.sh 0 ./data/custom
```

Prefer a validated `./data/custom/pocket.lmdb`. Output is
`./data/custom/pocket_reps.pkl`, containing `(pocket_names, pocket_reps)` with
shape `(number_of_pockets, 6, 128)` for the six-fold models. This is not the same
shape as the flattened arrays used by chunked screening. If the LMDB is absent,
the legacy helper attempts ligand-defined pocket extraction from PDB filenames;
read its limitations in [data formats](DATA_FORMAT.md) before using it.

## Benchmark

Download DUD-E or LIT-PCBA inputs from the same data repository. Expected paths:

```text
data/DUD-E/<target>/mols.lmdb
data/DUD-E/<target>/pocket.lmdb
data/lit_pcba/<target>/mols.lmdb
data/lit_pcba/<target>/pockets.lmdb
```

```bash
# Single-checkpoint mode (the previous effective behavior).
TASK=DUDE USE_FOLDS=false WEIGHT_PATH=./data/benchmark_weights/90.pt bash test.sh
# Explicit six-fold mode; no separate single-checkpoint file is required.
TASK=DUDE USE_FOLDS=true bash test.sh
```

Set `TASK=PCBA` for LIT-PCBA. `USE_FOLDS` is now honored by both tasks; historical
runs that silently forced single-fold mode are not directly comparable. The
benchmark iterates the target directories; there is no `DUDE_TARGET_FILTER` or
automatic external-active TSV protocol in this branch. DUD-E writes
`Holo_RealPocket_GenPack.csv` in the working directory and overwrites that file on
the next run; preserve it and console logs per run. `RESULTS_PATH` is forwarded
to Uni-Core but does not redirect that legacy task-level CSV.

## Large-library chunked screening

This is an advanced CUDA path with separate data requirements:

```bash
mkdir -p results/chunks
python utils/screening_chunk.py --gpu_num 1 \
  --mol_embs path/to/chunk0.npy path/to/chunk1.npy \
  --zscore_embs path/to/reference.npy --pocket_reps path/to/pockets.pkl \
  --batch_size 4 --output_dir results/chunks
python utils/retrieve_chunk.py --input_files 'results/chunks/merge1_*.pkl' \
  --mol_lmdb path/to/chunk0.lmdb path/to/chunk1.lmdb \
  --output_dir results/retrieved --num_threads 4
```

Quote the glob: `--input_files` accepts one pattern, expanded internally.
Use only the final merge generation (`merge1_*` for two chunks), not every
intermediate merge. Keep embedding/LMDB chunk order and row order identical.
The existing implementation requires at least 100,000 molecules per chunk;
it is not the small-example path. See the remaining format and resume caveats in
[troubleshooting](TROUBLESHOOTING.md).
