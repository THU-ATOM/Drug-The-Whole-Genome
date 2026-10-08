# DUD-E 5-Target 6-Fold Virtual Screening Test

Date: 2026-10-08
Branch: `refactor/deep-restructure`
Code: `unimol/tasks/drugclip.py` (`test_dude` / `test_dude_target_ensemble`), `test.sh`

---

## Goal

Evaluate the DrugCLIP model's virtual-screening enrichment on 5 targets. For each
target we score a pocket against a molecule library and check whether the model
ranks the real binders at the top.

## Active vs. inactive

For each target the molecule library is a mix of two groups:

- **Active (label = 1)** — the target's true binders, taken from the external TSV
  `/home/gaobowen/library_actives_per_target.tsv`.
- **Inactive / decoy (label = 0)** — a random sample of the ChemDiv library
  (`chemdiv_1640k_new.lmdb`, 1.64M molecules). With `background_ratio = 0.01`,
  ~16,481 decoys are drawn per target; any molecule that is an active is excluded.

The 5 targets and their active counts:

| DUD-E folder | TSV name (alias) | actives |
|------|------|------|
| adrb2 | ADRB2 | 262 |
| esr2  | ESR2  | 93 |
| mk01  | MAPK1 | 635 |
| mk10  | MAPK10 | 27 |
| xiap  | XIAP  | 27 |

## Data required

| Path | Size | Role |
|------|------|------|
| `data/dict_mol.txt`, `data/dict_pkt.txt` | 4K each | ligand / pocket token dictionaries |
| `/home/gaobowen/library_actives_per_target.tsv` | 128K | **actives** source (see format below) |
| `/drug/encoding_mols/chemdiv_1640k_new.lmdb` | 2.5G | **decoy** pool (1.64M molecules) |
| `data/model_weights/6_folds/fold_0.pt … fold_5.pt` | 1.2G × 6 | 6-fold ensemble weights |
| `data/benchmark_weights/dude_identity_90.pt` | 1.2G | initial `--path` weight (loaded once, then each fold overrides it) |
| `data/DUD-E/{adrb2,esr2,mk01,mk10,xiap}/pocket.lmdb` | ~30K each | per-target pocket (the only per-target input read) |

Notes:
- `data/DUD-E/<target>/mols.lmdb` is **not** read when the TSV is present (actives come from the TSV, decoys from ChemDiv), but the `data/DUD-E/<target>/` folder must exist since targets are discovered via `os.listdir`.
- Total footprint ≈ 11 GB (6×1.2G folds + 1.2G init weight + 2.5G ChemDiv); everything else is tiny.

### TSV format (`library_actives_per_target.tsv`)

Tab-separated, header row present. The loader only uses **column 1 `target`** and
**column 3 `smiles`**; the target name is mapped to the DUD-E folder via
`_library_target_alias` (e.g. MAPK1→mk01, MAPK10→mk10).

```
target	uniprot_id	smiles	pActivity	...
ADRB2	P07550	Cn1nc(C(=O)OCCCCNC[C@H](O)...	9
```

## How the test works (`test_dude_target_ensemble`)

1. Build the per-target library: actives + ~16,481 ChemDiv decoys.
2. For each of the 6 fold weights (`fold_0.pt … fold_5.pt`), encode pocket × molecules
   into a similarity matrix.
3. Average the 6 matrices → robust-normalize (median/MAD) → take the max score over pockets.
4. Compute AUC / BEDROC (α=80.5) / EF / RE with `cal_metrics`.
5. Average the metrics over the 5 targets.

## Reproduce

```bash
cd /drug/gaobowen/Drug-The-Whole-Genome
export DUDE_TARGET_FILTER="adrb2,esr2,mk01,mk10,xiap"   # test.sh defaults to "ace"
export CUDA_VISIBLE_DEVICES=1
bash test.sh
```

Relevant `test.sh` defaults: `use_chemdiv=1`, `background_ratio=0.01`, `use_folds=True`.
The active TSV is picked up automatically via `_resolve_dude_active_tsv`.

## Results

| target | AUC | BEDROC | EF@0.5% | EF@1% | EF@2% | EF@5% |
|------|------|--------|---------|-------|-------|-------|
| adrb2 | 0.9852 | 0.7583 | 63.14 | 48.31 | 32.62 | 16.93 |
| esr2  | 0.9820 | 0.8144 | 122.39 | 78.37 | 44.02 | 18.92 |
| mk01  | 0.7762 | 0.0770 | 1.25 | 1.72 | 1.81 | 2.71 |
| mk10  | 0.8278 | 0.2220 | 22.10 | 22.10 | 20.32 | 9.62 |
| xiap  | 0.9991 | 0.9404 | 191.53 | 95.76 | 49.87 | 19.99 |

**Mean over 5 targets:**
- AUC = **0.9141**
- BEDROC = **0.5624**
- EF@0.5% = 80.08 · EF@1% = **49.25** · EF@2% = 29.73 · EF@5% = 13.63
- RE@0.5% = 97.12 · RE@1% = 54.03 · RE@2% = 30.91

adrb2, esr2, xiap rank best (AUC 0.98–1.0, EF@1% 48–96); mk01 is the hardest
(AUC 0.78, EF@1% 1.72); mk10 is in between.
