# Refactor scope and validation

## Code changes

- Share model setup across retrieval, molecule encoding and benchmark entry points.
- Centralize ordered fold checkpoint/cache paths while retaining their filenames.
- Parse explicit booleans and honor fold selection for DUD-E and LIT-PCBA.
- Validate retrieval input files before model allocation; create output parents.
- Stop custom-library retrieval from writing into distributed caches.
- Keep the top-2% policy, with one result for nonempty libraries smaller than 50.
- Check cache row identity/order, embedding shape and finite values across folds.
- Validate HDF5 resume manifests and contiguous completion masks; flush embeddings
  before marking rows complete and preserve original exceptions on failure.
- Write per-target metrics and run metadata to unique benchmark directories.
- Select benchmark targets explicitly and record skipped/failed inputs.
- Report PDB parsing failures and require CUDA at GPU-only model setup.
- Disable gradient recording in standalone inference entry points.
- Add a 12-molecule example generator/checker and a Linux CI test workflow.
- Quote shell arguments, expose configuration, remove the implicit 30,000-row
  encoding cutoff, and use the actual dictionary directory for pocket encoding.

Model architectures, learned parameters, embedding normalization, six-fold MAD
formula, pocket maximum aggregation and benchmark metric formulas are unchanged.
Selecting folds now works, so that option intentionally changes which models run.

## Lightweight checks

```bash
python -m pip install numpy h5py lmdb rdkit
python -m unittest discover -s tests -v
bash tests/test_wrappers.sh
python -m compileall -q unimol utils examples
git diff --check
```

The unit tests cover explicit false parsing, fold layouts, cache/custom preflight,
input-overwrite rejection, normal-library ranking equivalence, small/tied/empty
libraries, model setup policy, both benchmark dispatch paths, missing DUD-E inputs,
and custom retrieval preserving caches while matching cached scores for identical
encoder outputs. Task methods are executed from their source AST with neural
encoder doubles to avoid importing Uni-Core or requiring a GPU. Shell tests
capture argv and check quoting, mode selection, range handling and invalid usage.

Validation completed locally: 20 tests, shell argument checks, Python compilation,
local documentation links and `git diff --check`. HDF5 interruption/resume tests
use real files with deterministic encoder outputs. Example generation uses real
RDKit/LMDB with a synthetic test pocket. Cache tests include a reordered fold;
report tests cover target selection, missing inputs and evaluation failure.
Real-checkpoint GPU inference and the Docker build remain untested.

## Model validation to run

Use the same input records, weight hashes, precision and seed on the base and
refactor commits. Compare cached retrieval and custom retrieval scores/rankings;
confirm distributed cache hashes stay unchanged. Run one pocket encoder example
and molecule encoder example, and inspect record counts and output shapes.

For benchmarking, compare single-checkpoint mode to the base's effective mode.
Validate six-fold mode separately; it was previously disabled internally. Record
target-level metrics, test inputs and the screening protocol. Have a colleague
follow QUICKSTART from a clean checkout and retain the environment and console log.

The five-target record on `docs/dude-5target-6fold-test` describes a custom
ChemDiv-background evaluation with an external active TSV. The current benchmark
reads the target LMDB files directly; reproducing that record requires its
corresponding data-preparation code and inputs.

Use the [small example](../examples/mini_screen/README.md) to capture reference
scores in the model environment. Further work includes a general-purpose molecule
preparation tool, a pocket-to-chunk converter, and the large-library chunk pipeline.
