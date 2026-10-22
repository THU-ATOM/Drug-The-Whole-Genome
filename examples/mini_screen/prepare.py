"""Build a 12-molecule screening example around a supplied, trusted pocket."""
import argparse
import json
from pathlib import Path
import pickle

import lmdb
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

SMILES = ("CCO", "CCCO", "CC(C)O", "CC(=O)O", "CCOC(C)=O", "c1ccccc1",
          "Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1", "CC(=O)Nc1ccccc1", "c1ccncc1", "C1CCCCC1")

def write_records(path, records):
    env = lmdb.open(str(path), subdir=False, map_size=16 * 1024 * 1024)
    try:
        with env.begin(write=True) as txn:
            for index, record in enumerate(records):
                txn.put(str(index).encode("ascii"), pickle.dumps(record))
    finally:
        env.close()


def prepare(pocket_path, output_dir, pocket_key="0"):
    env = lmdb.open(str(pocket_path), subdir=False, readonly=True, lock=False)
    try:
        with env.begin() as txn:
            value = txn.get(pocket_key.encode("ascii"))
            if value is None:
                raise ValueError(f"Pocket key {pocket_key!r} not found in {pocket_path}")
            pocket = pickle.loads(value)
    finally:
        env.close()
    atoms = list(pocket["pocket_atoms"])
    coordinates = np.asarray(pocket["pocket_coordinates"], dtype=np.float32)
    if not atoms or coordinates.shape != (len(atoms), 3) or not np.isfinite(coordinates).all():
        raise ValueError("Pocket coordinates must be finite and match the nonempty atom list")
    if not any(atom != "H" for atom in atoms):
        raise ValueError("Pocket must contain heavy atoms")
    name = pocket["pocket"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Pocket identifier must be a nonempty string")
    records = []
    for index, smiles in enumerate(SMILES):
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = 2026 + index
        if AllChem.EmbedMolecule(molecule, params) != 0:
            raise RuntimeError(f"Conformer generation failed for {smiles}")
        if AllChem.UFFOptimizeMolecule(molecule, maxIters=500) != 0:
            raise RuntimeError(f"Conformer optimization did not converge for {smiles}")
        molecule = Chem.RemoveHs(molecule)
        records.append({
            "atoms": [atom.GetSymbol() for atom in molecule.GetAtoms()],
            "coordinates": np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float32),
            "smi": Chem.MolToSmiles(molecule),
        })
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_records(output / "mols.lmdb", records)
    write_records(output / "pocket.lmdb", [{"pocket": name, "pocket_atoms": atoms,
                                          "pocket_coordinates": coordinates}])
    manifest = {"rdkit_version": rdBase.rdkitVersion, "conformer_seed_base": 2026,
                "pocket_source": str(Path(pocket_path).resolve()), "pocket_key": pocket_key,
                "molecule_count": len(records), "pocket_count": 1,
                "expected_result_rows": max(1, int(len(records) * .02)),
                "smiles": [r["smi"] for r in records]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(records)} molecules and one pocket in {output}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pocket-lmdb", type=Path, required=True)
    parser.add_argument("--pocket-key", default="0")
    parser.add_argument("--output-dir", type=Path, default=Path("results/mini"))
    args = parser.parse_args()
    prepare(args.pocket_lmdb, args.output_dir, args.pocket_key)
