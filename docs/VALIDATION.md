# Refactor scope and validation

## Code changes

- Share model setup across retrieval, molecule encoding and benchmark entry points.
- Centralize ordered fold checkpoint/cache paths while retaining their filenames.
- Parse explicit booleans and honor fold selection for DUD-E and LIT-PCBA.
- Validate retrieval input files before model allocation; create output parents.
- Stop custom-library retrieval from writing into distributed caches.
- Keep the top-2% policy, with one result for nonempty libraries smaller than 50.
- Keep DUD-E CSV target labels aligned with targets actually evaluated.
- Quote shell arguments, expose configuration, remove the implicit 30,000-row
  encoding cutoff, and use the actual dictionary directory for pocket encoding.

Model architectures, learned parameters, embedding normalization, six-fold MAD
formula, pocket maximum aggregation and benchmark metric formulas are unchanged.
Selecting folds now works, so that option intentionally changes which models run.

## Lightweight checks

```bash
python -m unittest discover -s tests -v
bash tests/test_wrappers.sh
python -m compileall -q unimol utils
git diff --check
```

The unit tests cover explicit false parsing, fold layouts, cache/custom preflight,
input-overwrite rejection, normal-library ranking equivalence, small/tied/empty
libraries, model setup policy, both benchmark dispatch paths, missing DUD-E inputs,
and custom retrieval preserving caches while matching cached scores for identical
encoder outputs. Task methods are executed from their source AST with neural
encoder doubles to avoid importing Uni-Core or requiring a GPU. Shell tests
capture argv and check quoting, mode selection, range handling and invalid usage.

These are software regression checks. They do not reproduce published metrics or
establish real-model numerical equivalence. No checkpoint/GPU run or Docker build
was performed for this refactor.

## Before declaring a tested model release

Use the same input records, weight hashes, precision and seed on the base and
refactor commits. Compare cached retrieval and custom retrieval scores/rankings;
confirm distributed cache hashes stay unchanged. Run one pocket encoder example
and molecule encoder example, and inspect record counts and output shapes.

For benchmarking, compare single-checkpoint mode to the base's effective mode.
Validate six-fold mode separately; it was previously disabled internally. Record
target-level metrics and test inputs rather than importing historical numbers
from a document that describes a different protocol. Have a colleague follow
QUICKSTART from a clean checkout and retain the environment and console log.

Remaining work includes a bundled scientific smoke fixture, a verified
SMILES/conformer writer, validated pocket-to-chunk conversion, Docker rebuilding,
and hardening the legacy large-library chunk/resume pipeline.
