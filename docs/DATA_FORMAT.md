# Input and output formats

These are contracts read by the current loaders, not a claim that arbitrary
SMILES or structures can be converted without chemical preprocessing.

## Molecule LMDB for retrieval and benchmark

Use a single LMDB file (`subdir=False`) with contiguous ASCII integer keys
`b"0"`, `b"1"`, ... and pickled Python dictionaries. Avoid metadata keys in this
legacy layout. Only load trusted LMDB/pickle files.

| Field | Expected content |
| --- | --- |
| `atoms` | Nonempty list/array of element symbols in coordinate order |
| `coordinates` | NumPy array `(N, 3)`, or a list/array of conformers each `(N, 3)` |
| `smi` or `smiles` | Molecule SMILES string; the loader prefers `smiles` if both are present |
| `label` | Benchmark binary label, 1 for active and 0 for background; defaults to 0 in retrieval |

Coordinates must be finite and match the atom count. Use Cartesian coordinates
in angstroms, consistent with PDB and molecular conformer inputs. The loader
samples a conformer with its seed, removes hydrogens and normalizes coordinates.
Provide NumPy coordinate arrays, not only nested Python lists: the loader accesses
`.shape` and `.astype`. A missing SMILES becomes an empty string in the legacy
loader, so validate it before encoding.

Document the standardization policy (salts, stereochemistry, protonation),
conformer generation method/seed, failures and retained molecule IDs with any
custom library. This refactor does not introduce a new SMILES-to-conformer writer.

## Pocket LMDB

The single-file key/value layout is the same. Each record contains:

| Field | Expected content |
| --- | --- |
| `pocket` | Stable pocket identifier |
| `pocket_atoms` | Nonempty atom element list/array |
| `pocket_coordinates` | Finite `(N, 3)` coordinates aligned with the atoms |

The loader removes hydrogens, crops to `--max-pocket-atoms`, and normalizes
coordinates. Record structure source, chain, pocket definition and residue set.
A full protein is not automatically a validated binding pocket.

The fallback in `unimol/encode_pockets.py` looks for ligand names between the
underscore and extension of PDB filenames (for example `example_LIG.pdb`). It
uses the first model and protein residues within 6 angstroms of ligand heavy
atoms. It can silently skip failed files and will reuse an existing `pocket.lmdb`.
Inspect records and counts; do not assume that directory processing succeeded
for every PDB or that a ligand-free AlphaFold structure supports this path.

## Encoding and cache formats are different

| Artifact | Reader or writer | Layout |
| --- | --- | --- |
| Retrieval fold cache | `retrieval_multi_folds` | Pickle pair `[mol_reps, mol_names]`, `(N, 128)` embeddings for one fold |
| Pocket encoder output | `encode_pockets.py` | Pickle pair `(pocket_names, pocket_reps)`, `(P, 6, 128)` |
| Molecule HDF5 | `encode_mols.py` | `mol_reps` with `(N, 768)` plus per-fold resume masks |
| Chunk molecule input | `screening_chunk.py` | NumPy array `(N, D)`; matching reference array has the same `D` |
| Chunk pocket input | `PocketDataset` | Pickle `(names, embeddings)` with a two-dimensional embedding array `(P, D)` |

Do not pass three-dimensional pocket encoder output straight into chunked
screening. Any conversion must preserve the same fold ordering, concatenation
and scaling for molecule and pocket embeddings. Shape compatibility alone does
not establish correct scores. This branch does not add a validated converter.

The directory LMDB v2 reader used by molecule encoding is a separate format with
named splits, including `success`; retrieval and benchmark use the single-file
loader. HDF5 resume masks track completed rows, not the identity of inputs or
weights. Use a fresh output location when either changes.

`utils/retrieve_chunk.py` requires `smiles` and `oid` fields and uses LMDB cursor
order to reconstruct rows. The legacy retrieval loader instead looks up numeric
keys by index. Those orders can differ (for example `10` sorts before `2` as text);
verify alignment explicitly before recovering IDs from chunk embeddings.

## Scoring

Retrieval normalizes each projected fold embedding, averages fold dot-product
matrices, and, for six-fold versions, standardizes each pocket's scores using
the candidate library median and median absolute deviation (MAD). It then takes
the maximum over pockets. Eight-fold retrieval omits that MAD step in the current
implementation. Chunked screening uses its explicit reference library for MAD.

The resulting scales depend on pipeline and background. Scores should not be
read as experimental affinities, probabilities, or directly comparable values
across unrelated reference libraries.
