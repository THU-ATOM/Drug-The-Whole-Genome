# Small screening example

This example generates 12 simple molecules with one RDKit ETKDGv3 conformer each,
optimized with UFF, and copies one pocket from a trusted prepared LMDB. It checks
the input/output path. The molecules are not a benchmark active/decoy set.

Use the environment in [installation](../../docs/INSTALL.md), the six-fold weights
from [quick start](../../docs/QUICKSTART.md), and a validated pocket from your data.
Run from the repository root:

```bash
python examples/mini_screen/prepare.py \
  --pocket-lmdb ./data/targets/YOUR_TARGET/pocket.lmdb \
  --output-dir ./results/mini
POCKET_PATH=./results/mini/pocket.lmdb MOL_PATH=./results/mini/mols.lmdb \
  USE_CACHE=false DEVICE=cuda SAVE_PATH=./results/mini/screen.csv bash retrieval.sh
python examples/mini_screen/check_result.py \
  --manifest ./results/mini/manifest.json --result ./results/mini/screen.csv
```

Choose an unused output directory when preparing another example. `--pocket-key`
defaults to `0`. Only load trusted LMDB files because their records contain pickle.
The manifest records the RDKit version, conformer seed, molecule identifiers and
expected counts. Keep it with the command, weight hashes and screening result.

Expected structure: 12 molecule records, one pocket record, and one result row
under the top-2% policy. A successful format check prints:

```text
Result checks passed: 1 row(s)
```

The winning molecule and score depend on the supplied pocket and checkpoints.
Real-checkpoint reference scores have not yet been recorded; current validation
covers input preparation and output checks. See [validation](../../docs/VALIDATION.md).
