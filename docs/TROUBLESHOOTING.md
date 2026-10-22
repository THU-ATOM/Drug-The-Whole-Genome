# Troubleshooting and common mistakes

## A command cannot find a file

Run from the repository root. The dictionary argument is `./dict`; the chunk
script is `utils/screening_chunk.py` (singular); dependencies are listed in
`docker/requirements.txt`. Check archive nesting and the exact target directory.
LIT-PCBA expects `data/lit_pcba/<target>/pockets.lmdb`, whereas DUD-E expects
`data/DUD-E/<target>/pocket.lmdb`.

Configure the wrappers with environment variables.
Use the uppercase names in [quick start](QUICKSTART.md). Quote paths with spaces.
Retrieval checks input paths before allocating the model and lists any missing files.

## Changing MOL_PATH did not change the screened library

`USE_CACHE=true` selects the distributed pre-encoded library. Use
`USE_CACHE=false` for a custom single-file LMDB. Custom retrieval leaves
`data/encoded_mol_embs/` unchanged. If earlier custom runs have
already modified those files, restore the intended cache archive before using
cached retrieval. Keep molecule identifiers and ordering consistent across folds.

## A cache reports a molecule-order or embedding error

Use caches produced for the same ordered molecule library and checkpoint set.
The error names the cache and, for an order mismatch, the first differing row.
Restore the matching archive or regenerate the complete set. Sorting names alone
would disconnect them from their embedding rows. NaN/infinite values require
checking the source coordinates and encoding run before regenerating embeddings.

## HDF5 resume is rejected

A changed manifest means the input, weights, dictionaries, preprocessing code or
encoding settings differ. Keep the earlier output and choose a new directory.
Older HDF5 files have no manifest and also require a fresh directory. A completion
mask with gaps indicates an inconsistent partial file; start a new output file.

## A benchmark skipped or failed a target

Open the run's `metrics.csv` or `summary.json` under `RESULTS_PATH`. Missing
`mols.lmdb` or the task-specific pocket file is recorded as skipped. An evaluation
error is recorded as failed and the command exits with the original traceback.
Correct the input and rerun with `TARGETS` set to the affected target names.

## Local scores differ from the website

Compare checkpoint version, fold count, pocket residues, structure coordinates,
conformers, candidate library, score normalization reference and aggregation.
Six-fold retrieval uses a median/MAD transformation, and its numerical scores
can change with the library. Compare the same molecules and pocket under matching
settings. Ask for the website's configuration if it is not available in its output.
See [issue #10](https://github.com/THU-ATOM/Drug-The-Whole-Genome/issues/10).

## USE_FOLDS appeared to have no effect

Set `USE_FOLDS=true` to load the six-fold weights, or `USE_FOLDS=false` to load
`WEIGHT_PATH`. The default is false. If the option has no effect in an older
checkout, update to a revision containing the fold-selection fix listed in
[validation](VALIDATION.md). Record the model mode when comparing runs.

## Small custom libraries produce no results

Ordinary retrieval returns the top 2%, with at least one result for a nonempty
library. Older versions rounded this count to zero when `N < 50`; update if you
encounter that behavior. For chunked screening, each chunk must contain at least
100,000 molecules.

## Pocket encoding reports a tensor dimension error

Record the exact commit, command, batch size, field shapes and traceback. Verify
the [pocket fields](DATA_FORMAT.md) and distinguish projected pocket embeddings
from per-atom hidden representations. Update older checkouts as suggested in
[issue #12](https://github.com/THU-ATOM/Drug-The-Whole-Genome/issues/12), then rerun
the failing input. If the error persists, include the field shapes and traceback
in your report; reducing batch size alone may leave the underlying mismatch.

## SMILES or AlphaFold structures are not accepted directly

Retrieval consumes prepared molecule and pocket LMDB files. Document preprocessing
and conversion, and validate a small sample before preparing a large library.
For a ligand-free structure, define or predict the pocket before converting it
to LMDB. See [issue #13](https://github.com/THU-ATOM/Drug-The-Whole-Genome/issues/13)
and [issue #2](https://github.com/THU-ATOM/Drug-The-Whole-Genome/issues/2).

## Chunked screening or resumed encoding gives unexpected output

- `--zscore_embs` supplies reference molecule embeddings to estimate each pocket's
  score median/MAD used in score normalization.
  If omitted, the first chunk is used. Choose a representative reference and keep
  it fixed across chunks. See [issue #5](https://github.com/THU-ATOM/Drug-The-Whole-Genome/issues/5).
- Preserve LMDB/embedding row order and chunk order during ID recovery. Feed only
  the last merge generation to `retrieve_chunk.py`.
- Use the matching two-dimensional pocket embedding format; raw encoder output
  has a fold axis and needs an appropriate conversion.
- The legacy chunk pipeline keeps chunk IDs as uint8 (do not exceed 256 chunks),
  assumes sufficiently large chunks, and may pad distributed samples across GPUs.
  Start with one GPU and inspect duplicate pocket names before scaling out.
- Use a new HDF5 output directory for changed inputs or encoding settings.
  See the resume checks above.

## Installation and platform errors

[Issue #11](https://github.com/THU-ATOM/Drug-The-Whole-Genome/issues/11) reports Conda
terms, `lsb_release`, and system-library conflicts during Docker builds. The
Dockerfile includes Conda terms commands, installs Conda under `/usr/local`, and
contains optional Docker installation steps. Match the failing build stage and
error message to the report in #11. Include the full build log when reporting a
new failure.

Match compiled Uni-Core extensions to the Python, PyTorch and CUDA versions.
Benchmarking and pocket encoding require CUDA. For a single GPU numbered 0, set
`CUDA_VISIBLE_DEVICES=0`.

## Reporting a problem

```text
Repository and commit:
OS, GPU, driver, Python, PyTorch, CUDA and Uni-Core versions:
Complete command:
Input type, field shapes, molecule/pocket counts:
Checkpoint set and fold version:
Cache use and background reference:
Expected output:
Actual output and complete traceback:
Small reproducible input, if shareable:
```
