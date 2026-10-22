# Quick start

Complete the [environment checks](INSTALL.md) first. All paths below are relative
to the repository root unless absolute. Set the target paths to match the
folders in your downloaded archive.

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
the `6_folds_filtered` cache directory. Use matching cache versions and the same
molecule order across folds. Retrieval checks the row counts, ordered identifiers,
embedding widths and finite values; mismatches identify the affected cache.

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
For the supplied 5HT2A configuration, select
`FOLD_VERSION=8_folds` and download matching inputs for that configuration.

Output is a headerless UTF-8 text file containing `smiles,score`, sorted by
descending score. It retains the top 2% (rounded down), with at least one result
for a nonempty library. See [Scoring](DATA_FORMAT.md#scoring) for the score definition.

## Screen a custom molecule library

Prepare a nonempty single-file molecule LMDB and a pocket LMDB following
[data formats](DATA_FORMAT.md). Then:

```bash
POCKET_PATH='./data/custom/pocket.lmdb' MOL_PATH='./data/custom/mols.lmdb' \
USE_CACHE=false SAVE_PATH='./results/custom.csv' \
DEVICE=cuda CUDA_VISIBLE_DEVICES=0 bash retrieval.sh
```

This path encodes the supplied molecules for each fold and leaves the distributed
caches unchanged.

The [small example](../examples/mini_screen/README.md) prepares 12 molecules,
copies one supplied pocket and checks the output count, identifiers and scores.

## Encode molecules separately

```bash
DEVICE=cuda bash encode_mols.sh 0 ./data/custom/mols.lmdb ./results/encoded
```

The wrapper writes HDF5 and processes the full input by default. For the directory
LMDB v2 format only, `START=0 END=1000` selects an index slice of the sorted unique
`success` keys. The single-file loader processes all records and logs that slicing
is ignored. For NumPy output use the Python
entry point with `--write-npy` (and optionally `--write-h5`). See its `--help`.

HDF5 files store an encoding manifest. Each run hashes the input LMDB, weights,
dictionaries and preprocessing code, then checks encoding settings and completion
masks before resuming. Hashing reads the input file once and can take time for a
large library. Keep input files unchanged during encoding. Use a new output
directory when these inputs/settings change or when an older HDF5 has no manifest.

## Encode pockets separately

```bash
bash encode_pocket.sh 0 ./data/custom
```

Prefer a validated `./data/custom/pocket.lmdb`. Output is
`./data/custom/pocket_reps.pkl`, containing `(pocket_names, pocket_reps)` with
shape `(number_of_pockets, 6, 128)` for the six-fold models. Chunked screening
requires a two-dimensional embedding array; see [data formats](DATA_FORMAT.md).
If the LMDB is absent, the PDB helper extracts ligand-defined pockets using the
filename convention described there.

## Benchmark

Download DUD-E or LIT-PCBA inputs from the same data repository. Expected paths:

```text
data/DUD-E/<target>/mols.lmdb
data/DUD-E/<target>/pocket.lmdb
data/lit_pcba/<target>/mols.lmdb
data/lit_pcba/<target>/pockets.lmdb
```

```bash
# Single-checkpoint mode.
TASK=DUDE USE_FOLDS=false WEIGHT_PATH=./data/benchmark_weights/90.pt bash test.sh
# Six-fold mode.
TASK=DUDE USE_FOLDS=true bash test.sh
```

Set `TASK=PCBA` for LIT-PCBA. Both tasks use `USE_FOLDS` to select the model mode.
For selected target directories, set `TARGETS` to space-separated names:

```bash
TASK=DUDE USE_FOLDS=true TARGETS='target_a target_b' \
RESULTS_PATH=./results/benchmarks bash test.sh
```

Replace those names with directories in your data. Omitting `TARGETS` evaluates
all target directories in sorted order. Unknown names fail before model loading.

Each run creates a unique `<task>-<UTC timestamp>-<id>/` under `RESULTS_PATH`:

- `metrics.csv`: per-target AUC, BEDROC, EF and RE at 0.5%, 1%, 2% and 5%, plus
  completion status and failure/skip reason.
- `summary.json`: means over completed targets, requested targets, status counts,
  model mode, CLI arguments and checkpoint SHA-256 hashes.

Missing target input files are recorded as skipped. Evaluation errors save the
report and stop the run with a traceback. A run with no completed targets fails.
Earlier reports remain in their own directories.

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
Use the final merge generation (`merge1_*` for two chunks). Keep embedding/LMDB
chunk order and row order identical. Each chunk requires at least 100,000
molecules. Use ordinary retrieval for smaller libraries. See
[troubleshooting](TROUBLESHOOTING.md) for format and resume details.
