# DrugCLIP for Drug-The-Whole-Genome

Encode molecular and protein-pocket structures and retrieve candidate molecules
with DrugCLIP. This repository includes multi-fold inference, benchmark entry
points and large-library screening utilities. It consumes prepared LMDB data;
SMILES or a full protein structure alone are not interchangeable with those inputs.

## Documentation

| Start here | Contents |
| --- | --- |
| [Installation](docs/INSTALL.md) | Environment recipe, import checks, CPU/GPU scope |
| [Quick start](docs/QUICKSTART.md) | Downloads, screening, encoding and benchmarks |
| [Data formats](docs/DATA_FORMAT.md) | LMDB fields, cache/embedding shapes, scoring |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common mistakes and existing issue guidance |
| [Applicability and reading](docs/METHOD_LIMITATIONS.md) | Independent analysis and interpretation limits |
| [Refactor validation](docs/VALIDATION.md) | Changes, regression checks and remaining model validation |

## Screen a prepared pocket

First follow the installation and download steps in [Quick start](docs/QUICKSTART.md).
Weights and pre-encoded molecule embeddings are distributed through
[DrugCLIP_data](https://huggingface.co/datasets/bgao95/DrugCLIP_data).
Run from the repository root and replace the pocket path with your actual file:

```bash
POCKET_PATH='./data/targets/YOUR_TARGET/pocket.lmdb' \
USE_CACHE=true DEVICE=cuda CUDA_VISIBLE_DEVICES=0 \
SAVE_PATH='./results/screen.csv' bash retrieval.sh
```

The default cached library is the distributed ChemDiv library (1,648,137 molecules).
For a custom molecule LMDB, set `USE_CACHE=false` and `MOL_PATH` explicitly.
Custom retrieval no longer overwrites distributed caches. Output is headerless
`smiles,score`, sorted by score, retaining the top 2% with at least one row for a
nonempty small library. The score is not an experimental affinity or probability.

## Benchmark

After preparing the benchmark data and appropriate weights:

```bash
TASK=DUDE USE_FOLDS=false bash test.sh
TASK=PCBA USE_FOLDS=true bash test.sh
```

`USE_FOLDS=false` uses `WEIGHT_PATH` (default `data/benchmark_weights/90.pt`).
`USE_FOLDS=true` now actually selects the six-fold ensemble; previous task methods
silently forced this option off. See [Quick start](docs/QUICKSTART.md#benchmark)
for exact input paths and output locations. Historical single-fold and six-fold
results should not be compared as the same protocol.

## Understanding DrugCLIP

Recommended independent reading: **Interrogating contrastive learning embeddings
for structure-based virtual screening: a case study on DrugCLIP**, Javier S.
Utgés, David T. Jones and Christine Orengo, bioRxiv preprint (2026).

[Paper](https://www.biorxiv.org/content/10.64898/2026.09.16.752029v1) ·
[Analysis code](https://github.com/JavierSanchez-Utges/DrugCLIP-pocket-analysis) ·
[Data](https://zenodo.org/records/22791467)

The study examines embedding geometry, generalization after overlap removal and
performance under different pocket sources. Read both its reported capabilities
and its limitations for apo/holo, AlphaFold and predicted pockets. Details and
interpretation cautions are in [Applicability](docs/METHOD_LIMITATIONS.md).

## Related tools

- [Original DrugCLIP repository](https://github.com/bowen-gao/DrugCLIP): check its
  own instructions and versions before transferring scripts or data between repos.
- [Pocket pretraining with ProFSA](https://github.com/THU-ATOM/ProFSA)
- [Virtual-screening post-processing](https://github.com/THU-ATOM/DrugCLIP_screen_pipeline)
- [Pocket detection](https://github.com/THU-ATOM/Pocket-Detection-of-DTWG)

## License

This project uses different licenses for different components:

- **Source Code**: Licensed under [Apache 2.0](LICENSE)
  - The source code is freely available for both academic and commercial use.

- **Database**: Licensed under [CC BY 4.0](docs/LICENSE.md)
  - The Drug-The-Whole-Genome database is freely available for both academic and commercial use with attribution.

- **Model Weights & Outputs**: Licensed under [CC BY-NC 4.0](docs/MODEL_WEIGHTS_LICENSE.md)
  - The DrugCLIP model weights and all results generated using the model are available for non-commercial use only.
  - For commercial use, please contact the authors for licensing options.

- **MIT Licensed Components**: [MIT License](unimol/LICENSE)
  - Part of the code is modified from Uni-Mol.
  - Copyright (c) 2022 DP Technology

Please see [LICENSE.md](docs/LICENSE.md) and [MODEL_WEIGHTS_LICENSE.md](docs/MODEL_WEIGHTS_LICENSE.md) for full details.

