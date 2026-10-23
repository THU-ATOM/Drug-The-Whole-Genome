# Advanced usage and source-level workflows

DrugCLIP ranks candidate molecules for a **specified protein binding pocket** by comparing learned three-dimensional representations of the pocket and molecules. Use it to prioritize a chemical library for further evaluation. Start with the supplied example, check your own input, and then run your target.

**You need a defined pocket and a compatible molecule library. A protein sequence, a whole-protein structure, or a SMILES list alone is not the prepared input to the screening command.** The pocket may come from an experimental complex, a previously extracted site, or a pocket-detection/refinement workflow such as GenPack.

The screening score is a ranking score. It is not a measured affinity, a binding pose, or evidence of experimental activity.

This guide describes the source at [commit `2bbe507`](https://github.com/THU-ATOM/Drug-The-Whole-Genome/tree/2bbe507deae4f37e11de4b2691328fd36539cb59). Commands below are Linux/Bash examples, run from the repository root unless stated otherwise. Match the documentation to your checkout when using another version.

Validation scope: small CPU component tests covered the LMDB validator, ligand-free serializer, reference holo extraction and selected score/output behaviors. The automated quick start additionally passed a dedicated inference-image build and a complete six-fold NET GPU run on an A100-SXM4-80GB. GenPack generation/refinement, end-to-end custom molecule-library/chunked workflows and published benchmark reproduction remain unverified.

For the automated six-fold demo and supported custom-pocket inputs, start with [the short README](../README.md). This page describes the underlying research scripts and additional workflows; they require manual setup.

For restricted networks, an alternative Python package index can be selected before
the automatic build with `export DRUGCLIP_PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`.
The Docker build also accepts `--build-arg PIP_INDEX_URL=URL`. These select the package
transport; dependency versions remain defined in `requirements-inference.txt`.
For a blocked Docker registry, pre-pull the same official PyTorch tag from an accessible
registry mirror, check its digest against the comment in `Dockerfile.inference`, and tag
that image with the official name before running the entry point. A warm image/cache
test should be reported separately from a fresh-download build.

## Contents

- [Choose a workflow](#choose-a-workflow)
- [Install and check the environment](#install-and-check-the-environment)
- [Download the matching resources](#download-the-matching-resources)
- [Run the supplied example](#run-the-supplied-example)
- [Prepare your own pocket](#prepare-your-own-pocket)
- [Validate a pocket LMDB](#validate-a-pocket-lmdb)
- [Screen your own molecule library](#screen-your-own-molecule-library)
- [Understand the scores and output](#understand-the-scores-and-output)
- [Large libraries and chunked screening](#large-libraries-and-chunked-screening)
- [Post-process and evaluate candidates](#post-process-and-evaluate-candidates)
- [Reproduce a result](#reproduce-a-result)
- [Troubleshooting](#troubleshooting)
- [Applicability and independent analysis](#applicability-and-independent-analysis)
- [Related tools and citations](#related-tools-and-citations)
- [Licensing](#licensing)

## Choose a workflow

```text
Experimental protein-ligand complex ── select the intended chains/ligand ──┐
Previously extracted pocket ────────── preserve and serialize its atoms ─┼─ pocket.lmdb
Apo/predicted protein ── locate a site; consider GenPack refinement ─────┘
                                                                         │
Matching checkpoints + supplied embeddings OR your prepared molecules ── DrugCLIP
                                                                         │
                                                        ranked candidates
                                                                         │
                                         identity checks, filters, docking,
                                         experimental follow-up as needed
```

| What you have | Start here | What to check |
|---|---|---|
| A supplied target LMDB | Run the supplied example | Matching model folds, molecule embeddings and dictionaries |
| A complex with a ligand in the intended site | Ligand-defined pocket extraction | Protein assembly/copy, chains, ligand residue code and ligand copy |
| A ligand-free pocket already extracted from a known complex | Serialize that pocket directly | Preserve its atoms and geometry; retain its extraction provenance |
| An apo structure, AlphaFold model, or newly predicted site | Pocket detection/refinement | Site localization and structure quality; consider the linked GenPack workflow |
| Only a sequence or SMILES strings | Prepare structure/conformers first | These are not replacements for pocket/molecule coordinate records |

Core screening can run locally or on your own server. AIPherDD and access to a particular supercomputer are not prerequisites for this repository. Larger hosted libraries or other services have their own access and input contracts.

## Install and check the environment

Clone the repository and record the version:

```bash
git clone https://github.com/THU-ATOM/Drug-The-Whole-Genome.git
cd Drug-The-Whole-Genome
git rev-parse HEAD
```

The provided [Dockerfile](../docker/Dockerfile) specifies Python 3.9, PyTorch 2.1.2 and CUDA 12.1, and builds Uni-Core. It needs an NVIDIA driver and NVIDIA Container Toolkit on the host for GPU use. The build downloads packages and is not a fully locked environment.

```bash
docker build -t drugclip:local -f docker/Dockerfile docker
docker run --rm -it --gpus '"device=0"' --ipc=host \
  -v "$PWD:/workspace" -w /workspace drugclip:local bash

# Inside the container; the current task module imports h5py.
python -m pip install h5py
```

Use `docker/` as the build context: the Dockerfile copies its `requirements.txt`. If you manage the environment yourself, install a compatible PyTorch/CUDA/Uni-Core combination and inspect [docker/requirements.txt](../docker/requirements.txt). Installing that requirements file alone does not install every prerequisite, notably Uni-Core and LMDB; `h5py` is also needed by the task module. Pin and record your final resolved versions.

Run this check **in the same environment that will run inference**:

```bash
python - <<'PY'
import sys
import numpy, torch, lmdb, Bio, rdkit, unicore, h5py
print("Python:", sys.version)
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("NumPy:", numpy.__version__)
PY
```

The pocket-encoding implementation moves samples to CUDA internally; do not assume `--cpu` makes that entry point CPU-compatible. The chunked utilities additionally require a PyTorch/CUDA-compatible installation of [`torch-scatter`](https://github.com/rusty1s/pytorch_scatter), which is not included in the requirements file. Inference memory and runtime depend on the number/size of pockets and molecules; no fixed hardware minimum or runtime is established by this guide.

## Download the matching resources

Official resources: [THU-ATOM/DrugCLIP_data](https://huggingface.co/datasets/THU-ATOM/DrugCLIP_data) on Hugging Face. The older `bgao95/DrugCLIP_data` link redirects to this dataset. Record the dataset revision and file hashes used in your run.

For the supplied cached example, obtain:

| Resource | Purpose |
|---|---|
| `model_weights.zip` | Fold-specific model checkpoints |
| `encoded_mol_embs.zip` | The matching supplied molecule embeddings and names |
| `WetLab_PDBs_and_LMDBs/NET/pocket.lmdb` | A ready-made example pocket ensemble |
| `targets.zip` | Alternative supplied target pockets; choose an actual target from the extracted archive |

Download the first three resources for the NET example, using the dataset's download interface or Hugging Face CLI. The two archives are approximately 13.0 GB and 5.9 GB compressed at the inspected resource revision; allow additional space for extraction. You do not need the benchmark archives for this example.

Inspect archive contents before extracting and arrange files so the following paths actually exist. Avoid an extra nested `data/data/` directory.

```text
Drug-The-Whole-Genome/
├── dict/
│   ├── dict_mol.txt
│   └── dict_pkt.txt
├── data/
│   ├── model_weights/6_folds/fold_0.pt ... fold_5.pt
│   ├── encoded_mol_embs/6_folds/fold0.pkl ... fold5.pkl
│   └── WetLab_PDBs_and_LMDBs/NET/pocket.lmdb
├── unimol/
└── runs/
```

The positional `./dict` argument used below points to the dictionaries. Checkpoints and cached embeddings are resolved separately under `./data`, relative to the working directory. A dictionary path and a data-asset directory are not interchangeable.

```bash
python - <<'PY'
from pathlib import Path
required = [Path("dict/dict_mol.txt"), Path("dict/dict_pkt.txt"),
            Path("data/WetLab_PDBs_and_LMDBs/NET/pocket.lmdb")]
required += [Path(f"data/model_weights/6_folds/fold_{i}.pt") for i in range(6)]
required += [Path(f"data/encoded_mol_embs/6_folds/fold{i}.pkl") for i in range(6)]
missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
assert not missing, "Missing or empty files: " + ", ".join(missing)
print("Required files exist; still validate LMDB contents and fold/cache identity.")
PY
```

## Run the supplied example

Start with the ready-made NET pocket and supplied **six-fold** molecule cache. This tests your installation and asset layout before introducing custom preprocessing.

```bash
mkdir -p runs/net
CUDA_VISIBLE_DEVICES=0 python unimol/retrieval.py ./dict \
  --user-dir ./unimol --valid-subset test \
  --num-workers 0 --ddp-backend=c10d --batch-size 4 \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --max-pocket-atoms 511 --seed 1 \
  --pocket-path data/WetLab_PDBs_and_LMDBs/NET/pocket.lmdb \
  --fold-version 6_folds --use-cache true \
  --save-path runs/net/results.txt
```

The output is a descending list of `molecule_name,score`, **without a header**. With the supplied cache, molecule names are intended to identify the supplied compounds; inspect the cache metadata before treating every name as a valid SMILES or supplier ID. Check nonempty output, finite scores and traceable molecule identities. Running successfully is an execution check, not a reproduction of a published enrichment or experimental hit rate.

For the original 5-HT2A setup, use the matching `8_folds` checkpoints **and** caches and set `--fold-version 8_folds`. There is no `--num-folds` flag on this retrieval entry point. Do not mix six-fold and eight-fold resources. See the score differences below.

The shell wrappers are examples with preset paths, devices and options. The explicit command above avoids inheriting their hardcoded sample paths and `--cpu` setting.

## Prepare your own pocket

### Route 1: use a known bound ligand to define the site

A holo complex contains a ligand in the relevant protein site. Identify the desired biological assembly, protein chains and ligand instance. A ligand residue code such as `DSP` is a file identifier, not a universal parameter called `LIG`. Different structures use different codes, and one structure may contain several copies of the same ligand.

Use a structure viewer or the structure's entity/chain records to confirm these identities. Export only the intended protein copy/chains and relevant bound ligand before extraction; the helper below does not offer a chain-selection argument. For an interface pocket, include both intended protein partners. Keep chain/residue mappings and essential cofactor information for later structural work.

The reference extractor is in [DrugCLIP_screen_pipeline](https://github.com/THU-ATOM/DrugCLIP_screen_pipeline). Clone that repository alongside this one:

```bash
git clone https://github.com/THU-ATOM/DrugCLIP_screen_pipeline.git ../DrugCLIP_screen_pipeline
mkdir -p runs/my_target
# Put the selected complex here, named target_<actual ligand residue code>.pdb.
# Example name only: target_DSP.pdb; DSP must really be present in that file.
python ../DrugCLIP_screen_pipeline/pocket_from_pdb.py \
  runs/my_target --name pocket
```

This helper retains **complete standard amino-acid residues** with a minimum atom-to-ligand-heavy-atom distance of **6 Å or less**, excludes ligand/non-protein residues from the output pocket, removes receptor hydrogens from the stored atom list, and writes `runs/my_target/pocket.lmdb`. Remove receptor hydrogens before extraction if you require a strictly heavy-atom distance definition: the helper's distance calculation itself includes any receptor hydrogens present.

The filename parser expects the part after the underscore and before `.pdb` to be the ligand code; use a simple prefix without additional underscores. Each matching ligand copy can create a pocket record. The helper merges protein chains internally and renumbers residues, so retain the original structure and your own mapping. It can silently skip invalid files: inspect the resulting database, not just the exit code. Use a fresh output directory rather than relying on an old LMDB being replaced cleanly.

For reproduction with released targets, retain the released LMDB. Re-extracting a supplied PDB with this separately maintained helper is not established to regenerate an identical pocket: compare atom identities/counts and coordinates before claiming equivalence. The supplied NET1 example and this helper's extraction produced different heavy-atom counts in the small preparation check.

### Already extracted, ligand-free pockets

An extracted pocket can legitimately contain no ligand: the original ligand was used to choose the protein atoms and then removed. Do not feed that file to a helper that searches for a ligand to define the site again. If its definition is already appropriate, preserve its coordinates and serialize it directly.

Save this example as `pocket_to_lmdb.py`. It expects a **previously selected pocket PDB**, not a complete protein with an unidentified site. It refuses to overwrite an existing database.

```python
import argparse
import pickle
from pathlib import Path
import lmdb
import numpy as np
from Bio.PDB import PDBParser, is_aa

p = argparse.ArgumentParser()
p.add_argument("pdb")
p.add_argument("output")
p.add_argument("--name", required=True)
a = p.parse_args()
out = Path(a.output)
assert not out.exists(), f"Output already exists: {out}"
model = PDBParser(QUIET=True).get_structure("pocket", a.pdb)[0]
atoms = [atom for residue in model.get_residues()
         if is_aa(residue, standard=True)
         for atom in residue.get_atoms() if atom.element != "H"]
xyz = np.asarray([atom.coord for atom in atoms], dtype=np.float32)
assert atoms and xyz.shape == (len(atoms), 3) and np.isfinite(xyz).all()
record = {"pocket": a.name,
          "pocket_atoms": [atom.element for atom in atoms],
          "pocket_coordinates": xyz}
out.parent.mkdir(parents=True, exist_ok=True)
env = lmdb.open(str(out), subdir=False, map_size=64 * 1024**2)
with env.begin(write=True) as txn:
    txn.put(b"0", pickle.dumps(record))
env.close()
print(f"Wrote one pocket with {len(atoms)} atoms; validate it before inference.")
```

```bash
python pocket_to_lmdb.py selected_pocket.pdb runs/my_target/pocket.lmdb \
  --name my_target
```

This serialization example selects standard amino acids and uses Biopython's selected alternate locations. Resolve alternate conformations and nonstandard residues explicitly when they matter. It does not establish the biological correctness of your chosen site.

### Route 2: apo or predicted structures and GenPack

If the site is not accurately localized or the structure comes from an apo/predicted receptor, inspect the [pocket detection and GenPack repository](https://github.com/THU-ATOM/Pocket-Detection-of-DTWG). Its generative route is:

1. Detect an initial candidate site with Fpocket, using a larger local region.
2. Use the backbone-conditioned SBDD/MolCRAFT-derived model to generate probe molecules in that region.
3. Pack/refine the protein-ligand complex.
4. Extract the local pocket around the refined probe, then serialize the pocket for DrugCLIP.

The generated molecule is used to help define the screening pocket; it is not automatically a synthesizable or validated hit. GenPack is an upstream preparation method, not a requirement to rerun on every high-quality ligand-defined pocket.

That repository documents [generation/refinement](https://github.com/THU-ATOM/Pocket-Detection-of-DTWG#pocket-detection-with-generative-model-refinement) and [template matching](https://github.com/THU-ATOM/Pocket-Detection-of-DTWG#pocket-detection-with-template-matching), with links to the generative model and checkpoint. **It currently contains research scripts with dataset-specific paths and chain conventions, rather than one portable end-to-end GenPack CLI.** Configure those paths and dependencies before running them. In particular, `SBDD_AUG/extract_pocket.py` assumes receptor chain `A` and probe chain `B` when no separate ligand file is supplied, and its default batch setup uses 128 worker processes. Do not run that batch entry point unchanged on an arbitrary machine or arbitrary chain IDs.

The GenPack extractor uses a strict distance comparison (`< threshold`), while the reference holo helper above uses `<= 6 Å`. Record which implementation you used. For benchmark reproduction, match the published preparation protocol, sampling and aggregation; do not select a generated pocket by performance on the evaluation labels.

## Validate a pocket LMDB

Each pocket record uses these fields:

| Field | Expected content |
|---|---|
| `pocket` | Nonempty identifier; use unique names for separately tracked sites |
| `pocket_atoms` | Ordered atom element symbols, one per coordinate row |
| `pocket_coordinates` | Finite coordinates in Å, shape `(N, 3)`, in the same order |

The database is a single-file LMDB (`subdir=False`) containing pickled records; the examples use ASCII record keys. Keep records free of the ligand used for extraction, water, and unrelated protein copies. The supplied pocket dictionary has `C`, `N`, `O`, `S`, `H` plus special tokens. Supply element symbols rather than PDB atom names such as `CA`. Metals and elements outside that vocabulary are not separately represented as those chemical elements by this encoder; record their role separately for downstream modeling.

The task removes hydrogens, centers coordinates and computes interatomic distances. Chain and residue identities are not separate model input features. Keep the original structure and mapping beside the LMDB.

**Set `--max-pocket-atoms 511` explicitly** to match the provided inference wrappers. The task parser's default is 256. Above the selected cap, preprocessing performs seeded, center-biased **atom sampling**, which can split residues. This is not a guarantee that a whole protein will be converted into the desired binding site. Compare atom counts before/after preprocessing and report any crop.

Save the following as `check_pocket_lmdb.py`. It checks the storage contract and conservatively rejects unknown element symbols; the model itself may otherwise map unsupported symbols rather than raise a clear error.

```python
import argparse
import pickle
import lmdb
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--max-atoms", type=int, default=511)
a = p.parse_args()
env = lmdb.open(a.path, subdir=False, readonly=True, lock=False,
                readahead=False, create=False)
seen = set()
with env.begin() as txn:
    assert txn.stat()["entries"] > 0, "Empty pocket LMDB"
    for key, value in txn.cursor():
        d = pickle.loads(value)
        name = d["pocket"]
        assert isinstance(name, str) and name.strip(), (key, "Missing pocket name")
        assert name not in seen, (key, "Duplicate pocket name", name)
        seen.add(name)
        atoms = np.asarray(d["pocket_atoms"])
        xyz = np.asarray(d["pocket_coordinates"], dtype=np.float32)
        assert atoms.ndim == 1 and len(atoms) > 0, (key, "Invalid atom list")
        assert xyz.shape == (len(atoms), 3), (key, "Atom/coordinate mismatch")
        assert np.isfinite(xyz).all(), (key, "Nonfinite coordinates")
        unknown = set(atoms.tolist()) - {"C", "N", "O", "S", "H"}
        assert not unknown, (key, "Unsupported element symbols", unknown)
        heavy = int(np.count_nonzero(atoms != "H"))
        assert heavy > 0, (key, "No heavy atoms")
        suffix = " WARNING: preprocessing will crop" if heavy > a.max_atoms else ""
        print(name, "atoms=", len(atoms), "heavy_atoms=", heavy, suffix)
env.close()
print("Storage checks passed; inspect the site and provenance separately.")
```

```bash
python check_pocket_lmdb.py data/WetLab_PDBs_and_LMDBs/NET/pocket.lmdb
python check_pocket_lmdb.py runs/my_target/pocket.lmdb --max-atoms 511
```

Then replace `--pocket-path` in the supplied screening command with your validated LMDB. Use a separate output directory. To compare individual pocket conformations, run them separately: the ordinary retrieval command combines all pocket records into one ranking.

## Screen your own molecule library

### Prepare molecules and preserve their identities

The ordinary uncached retrieval loader expects `atoms`, NumPy `coordinates`, and `smi`. A reliable representation is an element-symbol list plus a list of `(N, 3)` conformer arrays, and a SMILES string. Coordinates must describe those atoms in that order. Plain SMILES text must first be converted into chemically valid 3D conformers.

The reference [SDF2lmdb.py](https://github.com/THU-ATOM/DrugCLIP_screen_pipeline/blob/main/SDF2lmdb.py) reads SDF files, generates/optimizes conformers and stores `smi`, `atoms`, `coordinates`, `IDs`, and `subset`. It expects supplier identifiers in `IDNUMBER`, `Catalog ID`, or `ID` properties. It can discard conformer-generation failures, uses 32 workers by default, and does not fix a conformer-generation seed. Record failures and the final database hash if you use it; a repeated run is not guaranteed to regenerate identical conformers.

Use a **new output filename** for each conversion. The converter opens an existing LMDB and starts writing at key zero without clearing old records; a shorter rerun can leave stale records from the previous library. The example protects an existing output:

```bash
# Run from the DrugCLIP repository root; sdf_input should contain only SDF inputs.
mkdir -p runs/my_library
if [ -e runs/my_library/molecules.lmdb ]; then
  echo "Choose a fresh output path; existing molecule LMDB will not be cleared." >&2
else
  python ../DrugCLIP_screen_pipeline/SDF2lmdb.py \
    sdf_input runs/my_library/molecules.lmdb
fi
```

Validate counts, atom/coordinate correspondence, finite coordinates and molecule identities before encoding. Retain a mapping from every processed record to the supplier/source molecule, including rejected and duplicate records. Standardization, protonation, stereochemistry, conformer generation and chemical filters are choices to record, not invisible defaults. This inference code does not impose a universal molecular-weight or Lipinski cutoff that makes all remaining molecules reliable.

### Ordinary uncached retrieval

**Use an isolated checkout/working copy with its own `data/encoded_mol_embs/` directory.** In the inspected implementation, `--use-cache false` encodes your molecules but writes the resulting fold files to the same fixed cache paths used by supplied resources. Do not run this mode against caches you need to preserve.

```bash
# In the isolated working copy, with dictionaries/checkpoints already arranged.
mkdir -p data/encoded_mol_embs/6_folds runs/my_library
CUDA_VISIBLE_DEVICES=0 python unimol/retrieval.py ./dict \
  --user-dir ./unimol --valid-subset test \
  --num-workers 0 --ddp-backend=c10d --batch-size 4 \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --max-pocket-atoms 511 --seed 1 \
  --pocket-path runs/my_target/pocket.lmdb \
  --mol-path runs/my_library/molecules.lmdb \
  --fold-version 6_folds --use-cache false \
  --save-path runs/my_library/results.txt
```

For later cached runs, reuse those newly generated caches only with the same molecule library, row/name order, fold checkpoints and dictionaries. `--use-cache true` loads the cache; changing `--mol-path` does not replace it. Stock retrieval does not verify that the molecule names/order agree across all fold files, so check this yourself.

The molecule encoder samples a conformer when multiple conformers are available. Keep the seed, data ordering and preprocessing fixed when comparing runs. Different input conformers or library composition can change rankings and normalization.

## Understand the scores and output

For ordinary `retrieval.py`, the implementation:

1. Encodes each pocket and molecule with matching fold checkpoints and unit-normalizes each embedding.
2. Computes dot products and averages the similarities across folds.
3. For `6_folds` variants, computes a per-pocket modified z-score over the screened library: `0.6745 * (score - median) / (MAD + 1e-6)`, where MAD is median absolute deviation. The `8_folds` branch does not apply this step here.
4. Takes the **maximum across input pockets** for each molecule, sorts descending, and writes `int(number_of_library_records * 0.02)` rows.

The code therefore writes the **top 2%**, despite a nearby comment saying “top 1%”. A library with fewer than 50 records produces an empty result file under this cutoff. There is no top-k command-line option on this entry point. Use at least 50 records for an output-producing smoke run, and evaluate small-library experiments with an explicitly adapted, recorded selection rule when needed.

Modified z-scores depend on the background library. Scores from different libraries, pocket ensembles, normalization references, or six/eight-fold routes are not automatically comparable. A high score is a prioritization signal, not a binding-affinity unit or universal pass threshold. Adding more pockets changes the maximum-over-pockets selection and should be documented.

The output names come from the cache or `smi` field in the uncached loader. Two-column output does not retain supplier metadata by itself. Join identities back to your source table before selecting, purchasing, or reporting compounds.

## Large libraries and chunked screening

This is an **advanced route with a different data interface**, not a drop-in replacement for the ordinary retrieval command. First establish correctness with the smaller route. Inspect [screening_chunk.py](../utils/screening_chunk.py), [screening_utils.py](../utils/screening_utils.py) and [retrieve_chunk.py](../utils/retrieve_chunk.py) for your version.

### Encode and adapt the embedding layout

The six-fold encoders produce molecule arrays `(N, 6, 128)` and pocket pickles `(names, embeddings)` with embeddings `(P, 6, 128)`. The molecule CLI requires an existing output directory:

```bash
mkdir -p runs/chunked/encoded
CUDA_VISIBLE_DEVICES=0 python unimol/encode_mols.py ./dict \
  --user-dir ./unimol --valid-subset test \
  --num-workers 0 --ddp-backend=c10d --batch-size 64 \
  --task drugclip --loss in_batch_softmax --arch drugclip --seed 1 \
  --mol-path runs/my_library/molecules.lmdb \
  --save-dir runs/chunked/encoded --write-npy

CUDA_VISIBLE_DEVICES=0 python unimol/encode_pockets.py ./dict \
  --user-dir ./unimol --valid-subset test \
  --num-workers 0 --ddp-backend=c10d --batch-size 8 \
  --task drugclip --loss in_batch_softmax --arch drugclip \
  --max-pocket-atoms 511 --seed 1 --pocket-dir runs/my_target
```

For a single-file molecule LMDB, the current encoder ignores `--start/--end` subsetting; do not assume those options divide an ordinary LMDB into chunks. Directory-backed datasets use a different loader and split interface. The supplied `encode_mols.sh` has preset ranges and output flags; inspect it before reuse.

The chunked utility expects **two-dimensional** molecule and pocket vectors. For matched, unit-normalized six-fold embeddings, flattening the six fold blocks and dividing by `sqrt(6)` on both sides preserves the mean fold dot product. Apply exactly the same transformation to molecules, pockets and normalization-reference molecules. For example, for arrays that fit in memory:

```python
import pickle
from pathlib import Path
import numpy as np

def flatten_six(x):
    x = np.asarray(x, dtype=np.float32)
    assert x.ndim == 3 and x.shape[1:] == (6, 128), x.shape
    assert np.isfinite(x).all()
    assert np.allclose(np.linalg.norm(x, axis=-1), 1, atol=1e-4)
    return x.reshape(len(x), 768) / np.sqrt(np.float32(6))

Path("runs/chunked").mkdir(parents=True, exist_ok=True)
m = np.load("runs/chunked/encoded/mol_reps0None.npy")
np.save("runs/chunked/molecules_flat.npy", flatten_six(m))
with open("runs/my_target/pocket_reps.pkl", "rb") as f:
    names, p = pickle.load(f)
with open("runs/chunked/pockets_flat.pkl", "wb") as f:
    pickle.dump((names, flatten_six(p)), f)
```

`mol_reps0None.npy` reflects the current CLI defaults `start=0, end=None`; verify the actual emitted filename. This adapter is an example, not an existing repository command or a claim of equivalent end-to-end results. For truly large libraries, split/stream data and preserve the exact row map rather than loading the complete array as in this small adapter.

### Check the advanced route's current constraints

- Each molecule chunk needs **at least 100,000 rows**: `torch.topk(..., 100000)` is hardcoded. Use the ordinary route for a tiny example.
- Match the feature dimension and fold order of every molecule chunk, pocket and normalization reference. A six-fold 3D tensor or a per-fold `.pkl` cache is not a valid flat `.npy` chunk.
- Use a fixed representative normalization sample across chunks. The default is the first molecule chunk; record this choice. Its approximate normalization differs from normalization over the complete library.
- Install the compatible `torch-scatter` extension, create the output directory, and pass actual chunk paths. The code also derives metadata from path components, so prefer explicit paths such as `runs/chunked/chunk_000.npy` rather than a bare filename.
- `--gpu_num` starts that many GPU worker processes; workers assign device IDs themselves. Allocate the intended GPUs at the container/scheduler level and verify visibility rather than assuming an outer `CUDA_VISIBLE_DEVICES` restriction is honored unchanged.
- Chunk IDs are stored as `uint8` in this implementation. Keep indices within 0–255; more than 256 chunks requires a code change and validation.
- Pocket names are grouped after basename/extension normalization. Avoid different sites with colliding normalized names. Distributed sampling can repeat work when pockets do not divide evenly among workers; a one-GPU run avoids that source of repeated site outputs.

After preparing flat chunks and the fixed reference array:

```bash
mkdir -p runs/chunked/output
python utils/screening_chunk.py --gpu_num 1 \
  --mol_embs runs/chunked/chunk_000.npy runs/chunked/chunk_001.npy \
  --zscore_embs runs/chunked/reference_flat.npy \
  --pocket_reps runs/chunked/pockets_flat.pkl --batch_size 1 \
  --output_dir runs/chunked/output

# Two chunks are indexed 0 and 1. Read only the final merge1 files.
python utils/retrieve_chunk.py \
  --input_files 'runs/chunked/output/merge1_*.pkl' \
  --mol_lmdb runs/chunked/chunk_000.lmdb runs/chunked/chunk_001.lmdb \
  --output_dir runs/chunked/results --num_threads 1
```

Create the named chunks and reference array before using this template; the flattening example above does not automatically build them. Quote the merge glob because `--input_files` accepts one pattern string. Do not retrieve all intermediate merge stages into the same output directory: the script writes each pocket's CSV by name and later writes can replace earlier files.

The retrieval helper expects `smiles` and `oid` fields in each source molecule record, whereas the reference SDF converter writes `smi` and `IDs`. Adapt metadata in a separate compatible database if necessary. Its row order must match the rows actually encoded, including LMDB key iteration order; matching only the number of records is insufficient. Preserve the exact `--mol_embs` / `--mol_lmdb` chunk order. Output CSVs contain `smiles`, `oid`, and `score`, plus the written index column.

**Numerically re-sort exported chunk CSVs before selecting their first rows.** The current retrieval helper combines string metadata with scores in a NumPy array, coercing scores to text, then sorts that text column. For example, it can put `2.0` before `10.0` in descending order. Convert scores to finite numbers and write a separate corrected output, preserving compound IDs as strings:

```python
from pathlib import Path
import numpy as np
import pandas as pd

source = Path("runs/chunked/results")
output = Path("runs/chunked/results_numeric")
output.mkdir(parents=True, exist_ok=True)
files = sorted(source.glob("*.csv"))
assert files, "No retrieved CSV files found"
for path in files:
    frame = pd.read_csv(path, dtype={"smiles": str, "oid": str})
    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    assert np.isfinite(frame["score"].to_numpy()).all(), path
    frame.sort_values("score", ascending=False).to_csv(output / path.name, index=False)
```

Chunked GPU execution and the complete custom-library adapter path have not been end-to-end validated by this documentation update. Treat them as advanced, source-checked instructions requiring a local identity/score validation before scaling.

## Post-process and evaluate candidates

The linked [reference screening pipeline](https://github.com/THU-ATOM/DrugCLIP_screen_pipeline) covers novelty filtering, clustering, docking and optional pocket recycling. The broad sequence is:

1. Resolve each hit to a unique source molecule; inspect duplicates, stereochemistry, availability and your selected chemical-property filters.
2. If novelty is an objective, compare against an appropriately defined active set and record the database version, endpoints and similarity thresholds.
3. Cluster for diversity and select a tractable number of representatives.
4. Dock selected candidates into an appropriate prepared receptor; inspect poses, clashes and target-specific interactions.
5. Prioritize follow-up experiments and retain negative/positive controls appropriate to the claim being tested.

That pipeline consumes a **header followed by four fields in order**: molecule ID, supplier/subset, DrugCLIP score, SMILES. Ordinary core retrieval output has only two fields and no header. Perform an explicit join to the library's identity table; do not add invented IDs or assume the formats are interchangeable.

Its clustering and Glide scripts require a licensed Schrödinger installation and currently hardcode `SCHRODINGER = '/opt/schrodinger2021-2'`. Configure the copies you use for your installation; exporting an environment variable alone does not override that Python constant. Its documentation includes command templates and cofactor/grid options. It is a reference workflow to configure, not a dependency automatically installed by this repository.

The reference pipeline suggests score cutoffs in its own context. They are not universal activity thresholds. Preserve essential metal/cofactor information for docking where relevant; removing a component from the atom-token pocket representation does not make it irrelevant to binding chemistry. Recycling pockets from docking results changes the screening protocol: record iterations and avoid tuning them on evaluation labels.

## Reproduce a result

For each run, save:

- Code commit, environment/package versions, checkpoint/dictionary hashes and dataset revision.
- Original structure, selected assembly/copy/chains/ligand, coordinate source, pocket extraction rule and resulting atom/residue counts.
- Whether GenPack was used, its checkpoint/configuration, sampling counts/seeds and the rule for selecting or aggregating its outputs.
- Pocket LMDB hash and record names/counts; hydrogen/alternate-location choices and any atom sampling/cropping.
- Molecule library version/hash, identity mapping, standardization/conformer settings, failures, duplicate handling and row order.
- Model fold order, molecule cache identity, normalization background, pocket aggregation, output cutoff and exact command/log.
- Evaluation dataset/split, positive and negative controls, metric definition and any excluded/failed cases.

Reproducibility has separate levels: a command completes; input/output identities are correct; repeated execution is consistent; and the specified scientific evaluation is reproduced. Evidence for one level does not establish the next. A fixed seed does not repair a wrong pocket or cache, and identical seeds do not guarantee bitwise equality across software/hardware versions.

For the supplied benchmarks, download the matching `DUD-E.zip`, `LIT-PCBA.zip` and benchmark checkpoints, then inspect [test.sh](../test.sh). It sets `TASK` to `DUDE` or `PCBA`, uses `./data/benchmark_weights/90.pt` and enables fold testing. Benchmark labels/data layouts differ from the custom-screening schemas above; use the benchmark route to reproduce its metrics, rather than treating a ranked candidate file as a benchmark result.

## Troubleshooting

| Symptom | What to check first |
|---|---|
| Missing dictionaries, checkpoints or cache files | Repository-root working directory, `./dict` positional argument, extracted `./data` paths and matching fold resources |
| `ModuleNotFoundError` or compiled-extension error | Same Python environment for install/run; Uni-Core, LMDB, h5py; matching PyTorch/CUDA and `torch-scatter` for the chunked route |
| CUDA unavailable or unexpected CPU execution | GPU visibility inside the container, host driver/toolkit, wrapper `--cpu` flag; the pocket encoder has a CUDA-specific implementation |
| PDB conversion exits but LMDB is empty/partial | Actual ligand code and presence, filename parsing, selected assembly/chains, broad exception handling in the helper; count/read back records |
| Editing a PDB has no effect | `encode_pockets.py` reuses an existing `pocket.lmdb`; use a fresh run directory with a deliberately prepared input |
| Coordinates or tokens are invalid | Correct field names, `(N,3)` finite arrays, atom/coordinate order, element symbols and dictionary compatibility |
| Whole protein or oversized site gives unstable/unhelpful ranks | Define the actual site first; inspect cropping and before/after atom counts; do not use cropping as pocket detection |
| Output file is empty although execution completed | Ordinary retrieval writes `int(N*0.02)` rows, so fewer than 50 molecule records yields zero |
| Changing the molecule path has no effect | Cached mode uses the fixed cache files; verify cache identity and `--use-cache` |
| Scores/names differ unexpectedly across folds | Same checkpoints, dictionaries, molecule library/name order, pocket records, conformers, seed and normalization reference |
| GPU out of memory despite lowering CLI batch size | Some task loaders use hardcoded internal batch sizes; reduce the number of pocket records per run or investigate that version's actual loader |
| Chunk script missing or input dimension/top-k error | Correct filename `utils/screening_chunk.py`; flat 2D compatible vectors; at least 100,000 molecules in every chunk |
| Hits map to the wrong compounds | Exact chunk order and encoded row/key order; `smiles`/`oid` versus `smi`/`IDs`; compare several independently traced rows |
| Rerunning molecule conversion leaves unexpected records | The SDF converter does not clear an existing LMDB; use a fresh output and reconcile counts against the input/rejection log |
| Chunk CSV first rows are not the highest numeric scores | Apply the finite numeric conversion and re-sort above; the current CSV helper can sort score text lexicographically |
| Good execution but weak enrichment or implausible candidates | First check preparation/identity, then assess pocket accuracy, library composition, chemistry shift and model limitations; poor performance can remain after correct setup |

When asking for help, provide the commit, exact command, environment summary, error/log, pocket source and extraction rule, LMDB schema/counts, fold/cache versions and expected versus observed behavior. A small shareable example is more diagnostic than a screenshot of a score table.

## Applicability and independent analysis

Pocket localization and residue membership matter. A predicted site, an apo structure and a ligand-defined experimental pocket are different inputs even when they belong to the same protein. Good file formatting does not establish that the binding site is correct, and GenPack does not guarantee reliable rankings for every target or chemistry.

The independent preprint **“Interrogating contrastive learning embeddings for structure-based virtual screening: a case study on DrugCLIP”**, by Javier S. Utgés, David T. Jones and Christine Orengo, analyzes pocket/ligand embeddings, pocket-source perturbations and generalization. Its author-deposited abstract reports that, in its de-leakage benchmark, the bound ligand was recovered in the top 1% of 50,000 candidates for 55–75% of novel targets. That is the study's retrieval setting, not a prospective hit rate for any submitted protein.

The same abstract reports reduced performance across more realistic structure/pocket settings and identifies side-chain orientation and residue mismatch as contributing factors. Read the [preprint](https://doi.org/10.64898/2026.09.16.752029), [analysis code](https://github.com/JavierSanchez-Utges/DrugCLIP-pocket-analysis) and [data](https://zenodo.org/records/22791467) for the experimental design and limits. These findings support discussing both capabilities and failure modes; they do not establish that every reproduction failure is an input mistake.

Check difficult/new use cases with appropriate controls and a predefined evaluation. Docking or a high retrieval rank alone does not demonstrate affinity, selectivity, molecular-glue action or another target-specific mechanism.

## Related tools and citations

- [DrugCLIP paper](https://doi.org/10.1126/science.ads9530): Jia et al., *Deep contrastive learning enables genome-wide virtual screening*, Science, 2026.
- [Original DrugCLIP framework](https://arxiv.org/abs/2310.06367): Gao et al., *DrugCLIP: Contrastive Protein-Molecule Representation Learning for Virtual Screening*, NeurIPS, 2023.
- [Pocket detection / GenPack](https://github.com/THU-ATOM/Pocket-Detection-of-DTWG): upstream pocket localization and refinement.
- [DrugCLIP_screen_pipeline](https://github.com/THU-ATOM/DrugCLIP_screen_pipeline): pocket/molecule preparation and reference post-processing/docking.
- [ProFSA](https://github.com/THU-ATOM/ProFSA): pocket pretraining; not an extra training step required for running the released DrugCLIP checkpoints.
- [Drug The Whole Genome website](https://drug-the-whole-genome.yanyanlan.com/): supplied genome-scale screening resources.

The core screening repository does not itself bundle a trillion-molecule library or a DiskANN/supercomputer submission service. The supplied ChemDiv cache and separately hosted large-scale services are distinct resources. Document the actual library and search method used in a run.

## Licensing

Keep the repository's existing licensing files with your installation:

- [Root LICENSE](../LICENSE) describes source code as Apache 2.0 and the database as CC BY 4.0.
- [Model weights and outputs license](../MODEL_WEIGHTS_LICENSE) and its [detailed copy](../docs/MODEL_WEIGHTS_LICENSE.md) specify CC BY-NC 4.0 for weights and model-generated outputs, including precomputed embeddings.
- Uni-Mol-derived components retain their [MIT license](../unimol/LICENSE); dependencies and external docking tools have their own terms.

There is an existing source-code licensing discrepancy: [docs/LICENSE.md](../docs/LICENSE.md) describes source code as CC BY 4.0, whereas the root LICENSE says Apache 2.0. Maintainers should reconcile those files. This guide reports the discrepancy without changing either license.
