# DrugCLIP: start with a working example

DrugCLIP ranks molecules against a **defined 3D protein pocket**. Start with the released
NET example below, then supply your own pocket on your own Linux machine.
A rank is a screening prediction; it does not establish binding or activity.

## 1. Prerequisites

- A Linux machine with Git, Python 3.9+, Docker, an NVIDIA GPU and NVIDIA Container Toolkit.
- Permission to run Docker, and an idle GPU visible in `nvidia-smi`.
- Internet access to Docker Hub, GitHub, PyPI and Hugging Face on the first run.
- Space for approximately 19 GB of official downloads, extracted six-fold assets,
  the Docker image and outputs. Reserve at least 50 GB of free disk space.
- Validated on an A100-SXM4-80GB. GPU/RAM requirements depend on the workload; no minimum is established here.

The entry point builds its inference environment automatically using
[Dockerfile.inference](docker/Dockerfile.inference) and [requirements-inference.txt](requirements-inference.txt).
It uses PyTorch 2.1.2 / CUDA 12.1 and a pinned Uni-Core commit. Host Python needs no extra packages.

## 2. Run the official NET example

```bash
git clone https://github.com/THU-ATOM/Drug-The-Whole-Genome.git
cd Drug-The-Whole-Genome
python3 run_drugclip.py demo --device 0
```

Select an idle GPU with `--device INDEX` or `--device GPU-UUID`. The first run downloads
the official weights, ChemDiv embeddings and NET structures, checks SHA256 identities,
and loads **six matched folds**. Subsequent runs reuse the verified environment/cache.
The pinned release and expected file hashes are in [inference-assets.json](inference-assets.json).
The released demo has 10 pockets and 1,648,137 molecule records, giving 32,962 ranked rows.

Read `runs/net/results.csv` (`rank,molecule,score`). The released engine returns the
top 2% of the cached molecule records, after averaging the six folds, applying robust
per-pocket normalization and taking the maximum across supplied pockets/templates.
This score depends on the background library; it is not an affinity or a universal activity threshold.
The `molecule` cell preserves the complete cached identity (`ID,SMILES` in this release).
Its embedded comma is CSV-quoted; read this file with a CSV parser.

Each run also saves `run.log`, `run_manifest.json`, `inference_manifest.json`, the
prepared pocket LMDB and the unmodified raw ranking. Manifests record hashes,
fold loading, library identity/order, atom counts, runtime and output checks.
Existing output directories are protected: rerun with `--out runs/net_repeat`.

## 3. Screen your own defined pocket

Choose **one** input form. These commands use the same released molecule library:

```bash
# A single-file LMDB, including multiple named pocket templates if needed
python3 run_drugclip.py screen --pocket-lmdb /path/pocket.lmdb --out runs/my_lmdb

# An already extracted, ligand-free protein pocket PDB
python3 run_drugclip.py screen --pocket-pdb /path/pocket.pdb --out runs/my_pdb

# A complex with an actual site-defining ligand; select protein chains explicitly
python3 run_drugclip.py screen --complex runs/net/NET1_DSP.pdb \
  --chains A --ligand DSP --ligand-chain A --ligand-residue 701 --out runs/my_complex
```

Add `--device` to any command when using a GPU other than 0. For a complex, the tool
retains whole standard protein residues having an atom within 6 Å of the selected
ligand's heavy atoms. `--chains A,B` retains selected protein chains; ligand chain/number
disambiguate repeated ligand instances. Inspect `prepared_pocket.pdb` and
`pocket_definition.json` before interpreting a custom result.
As in the reference extractor, stored protein hydrogens can affect residue selection; only protein heavy atoms are serialized.

Input requirements:

- A pocket PDB must already identify the site; a whole protein is not a pocket.
  Use a single PDB model. Hydrogen and water atoms are excluded; the selected
  alternate atom location follows Bio.PDB's occupancy selection.
- LMDB records need unique nonempty `pocket` names, `pocket_atoms` element symbols
  and matching finite `pocket_coordinates` arrays of shape `(N,3)`, in ångströms.
  Elements are `C,N,O,S,H`; each record must contain 1–511 heavy atoms.
- The quick start rejects larger/unsupported inputs instead of silently cropping
  them. It preserves supplied coordinates and does not infer biological assemblies,
  repair missing residues, add a ligand or decide which binding site is correct.
- LMDB and embedding caches contain Python pickle data: use trusted sources.

## 4. Limits, preparation and troubleshooting

| What happened? | What to do |
|---|---|
| GPU unavailable/busy | Check `nvidia-smi`, Docker GPU support and an idle `--device`. |
| GPU out of memory | The released loader batches up to 16 pockets internally. Use fewer custom templates or a larger GPU. |
| Uni-Core says `fused_* is not installed corrected` | Optional fused extensions are not built here; PyTorch fallbacks are used. Check the final run status. |
| Build/import failure | Keep the build output; use the declared Dockerfile, not an unrelated Python environment. |
| Download interrupted | Rerun; partial downloads resume. Use `--cache-dir /path/cache` to share verified assets. |
| Download source unreachable | A compatible mirror can be set with `--endpoint URL`; hashes must still match. Alternatively transfer the pinned files into `CACHE/REVISION/` and pass `--cache-dir CACHE`. |
| Hash mismatch | Preserve the error, replace the corrupt file from the pinned official release, then retry. |
| Output already exists | Choose a fresh `--out`; previous results are never cleared. |
| Missing/ambiguous ligand | Check the actual residue code, chain and residue number in the PDB. |
| Oversized pocket or poor ranking | Verify site localization/residue membership and input preparation; correct execution does not guarantee enrichment. |

Apo/AlphaFold structures and uncertain sites need a separate preparation workflow.
See [pocket detection / GenPack](https://github.com/THU-ATOM/Pocket-Detection-of-DTWG).
GenPack is an independent upstream entry point; this quick start does not run it automatically.
The supplied NET LMDB and a newly extracted NET complex pocket need not contain identical atoms.

For custom molecule libraries, chunked screening, docking, benchmark reproduction,
input validators and interpretation, see [advanced usage](docs/advanced_usage.md).
An [independent analysis](https://doi.org/10.64898/2026.09.16.752029) examines capabilities
and sensitivity to pocket preparation; a failed run alone does not identify its cause.

## Citation and licenses

[Jia et al., *Deep contrastive learning enables genome-wide virtual screening*, Science (2026)](https://doi.org/10.1126/science.ads9530).
See [source/database licenses](LICENSE), [model weights and outputs license](MODEL_WEIGHTS_LICENSE)
and [component/license details](docs/advanced_usage.md#licensing).
Weights and model-generated outputs are under CC BY-NC 4.0.
