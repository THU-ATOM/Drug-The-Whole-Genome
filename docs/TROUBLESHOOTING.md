# Troubleshooting and common mistakes

## A command cannot find a file

Run from the repository root. The dictionary argument is `./dict`; the chunk
script is `utils/screening_chunk.py` (singular); dependencies are listed in
`docker/requirements.txt`. Check archive nesting and the exact target directory.
LIT-PCBA expects `data/lit_pcba/<target>/pockets.lmdb`, whereas DUD-E expects
`data/DUD-E/<target>/pocket.lmdb`.

The wrappers accept environment settings rather than requiring source edits.
Use the uppercase names in [quick start](QUICKSTART.md). Quote paths with spaces.
New retrieval preflight errors list missing weights, caches and inputs together.

## Changing MOL_PATH did not change the screened library

`USE_CACHE=true` selects the distributed pre-encoded library. Use
`USE_CACHE=false` for a custom single-file LMDB. In this branch custom retrieval
no longer overwrites `data/encoded_mol_embs/` caches. If earlier custom runs have
already modified those files, restore the intended cache archive before using
cached retrieval. Keep molecule identifiers and ordering consistent across folds.

## Local scores differ from the website

Compare checkpoint version, fold count, pocket residues, structure coordinates,
conformers, candidate library, score normalization reference and aggregation.
Six-fold retrieval uses a median/MAD transformation, and its numerical scores
can change with the library. A score near zero is not itself evidence of no
binding, and a large score is not a binding probability. The website's exact
configuration must be established before assigning a cause.
See [issue #10](https://github.com/THU-ATOM/Drug-The-Whole-Genome/issues/10).

## USE_FOLDS appeared to have no effect

Earlier benchmark task methods forced `use_folds=False` internally. This branch
removes those overrides and parses `true` and `false` as booleans. The wrapper and
CLI default to false to preserve the previous effective single-checkpoint mode.
`USE_FOLDS=true` now actually loads six fold weights; it does not load a separate
single-model checkpoint first. Compare results only under the same protocol.

The historical five-target document on another branch describes additional
ChemDiv/background/active-TSV processing not supplied here. It is not a validated
recipe for this branch, and its numbers are not regression expectations.

## Small custom libraries produce no results

Previously `int(N * 0.02)` selected zero molecules when `N < 50`. This branch
returns at least one result for a nonempty library. Larger libraries retain the
existing top-2% policy. This fix applies to ordinary retrieval; the advanced
chunk pipeline still hard-codes top-100,000 and needs at least that many molecules
in each chunk.

## Pocket encoding reports a tensor dimension error

Record the exact commit, command, batch size, field shapes and traceback. Verify
the [pocket fields](DATA_FORMAT.md) and distinguish projected pocket embeddings
from per-atom hidden representations. Lowering batch size is not a general fix:
the user in [issue #12](https://github.com/THU-ATOM/Drug-The-Whole-Genome/issues/12)
reported another error after doing so. The maintainer suggested updating the code;
verify the updated behavior with the actual input before treating the issue as fixed.

## SMILES or AlphaFold structures are not accepted directly

Retrieval consumes prepared molecule and pocket LMDB files. Document preprocessing
and conversion, and validate a small sample before preparing a large library.
The existing ligand-based PDB helper is not a general pocket predictor for
ligand-free structures. See [issue #13](https://github.com/THU-ATOM/Drug-The-Whole-Genome/issues/13)
and [issue #2](https://github.com/THU-ATOM/Drug-The-Whole-Genome/issues/2).

## Chunked screening or resumed encoding gives unexpected output

- `--zscore_embs` supplies reference molecule embeddings to estimate each pocket's
  score median/MAD. It standardizes scores, not the embedding distribution itself.
  If omitted, the first chunk is used. Choose a representative reference and keep
  it fixed across chunks. See [issue #5](https://github.com/THU-ATOM/Drug-The-Whole-Genome/issues/5).
- Preserve LMDB/embedding row order and chunk order during ID recovery. Feed only
  the last merge generation to `retrieve_chunk.py`.
- Use the matching two-dimensional pocket embedding format; raw encoder output
  has a fold axis and needs an appropriate conversion.
- The legacy chunk pipeline keeps chunk IDs as uint8 (do not exceed 256 chunks),
  assumes sufficiently large chunks, and may pad distributed samples across GPUs.
  Start with one GPU and inspect duplicate pocket names before scaling out.
- Use a new HDF5 output path for a new molecule set, ordering or model version.
  Resume flags are not a provenance check.

These advanced chunk/resume behaviors are documented limitations, not fixes
claimed by this refactor.

## Installation and platform errors

[Issue #11](https://github.com/THU-ATOM/Drug-The-Whole-Genome/issues/11) reports Conda
terms, `lsb_release`, and system-library conflicts during Docker builds. The
current Dockerfile already includes Conda terms commands; do not assume that
repeating that step addresses every failure. Record the build stage and full log.
The recipe still installs Conda under `/usr/local` and contains optional Docker
installation steps; this branch does not claim those reports are all resolved.

Keep compiled Uni-Core extensions compatible with the environment. GPU tasks do
not automatically support Apple Metal or CPU execution. For a single GPU numbered
0, use `CUDA_VISIBLE_DEVICES=0`, not the old hard-coded value 1.

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

Separate input/installation failures from methodological limitations. See
[recommended reading and applicability](METHOD_LIMITATIONS.md).
