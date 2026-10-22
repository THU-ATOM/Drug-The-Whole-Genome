"""Array checks and encoding provenance, independent of the model runtime."""
import hashlib
import json
from pathlib import Path

import numpy as np


def validate_embeddings(reps, names, source, reference_names=None, dimension=None):
    reps = np.asarray(reps)
    names = list(names)
    if reps.ndim != 2 or not all(reps.shape):
        raise ValueError(f"{source}: expected a nonempty 2D embedding array, got {reps.shape}")
    if len(names) != len(reps):
        raise ValueError(f"{source}: {len(names)} names for {len(reps)} embedding rows")
    if not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError(f"{source}: molecule identifiers must be nonempty strings")
    if dimension is not None and reps.shape[1] != dimension:
        raise ValueError(f"{source}: embedding width {reps.shape[1]} differs from {dimension}")
    if not np.issubdtype(reps.dtype, np.floating) or not np.isfinite(reps).all():
        raise ValueError(f"{source}: embeddings must be finite floating-point values")
    if reference_names is not None and names != reference_names:
        if len(names) != len(reference_names):
            detail = f"row count {len(names)} differs from {len(reference_names)}"
        else:
            row = next(i for i, (a, b) in enumerate(zip(names, reference_names)) if a != b)
            detail = f"row {row}: {names[row]!r} differs from {reference_names[row]!r}"
        raise ValueError(f"{source}: molecule order differs from the first fold ({detail})")
    return names


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoding_manifest(mol_path, checkpoints, settings, dictionaries):
    """Hash input bytes (including row order), weights, dictionaries and loader code."""
    path = Path(mol_path)
    # LMDB v2 stores both records and split definitions in data.mdb; lock.mdb
    # contains mutable reader state and is not part of the encoding input.
    data_file = path / "data.mdb" if path.is_dir() else path
    root = Path(__file__).resolve().parent
    sources = sorted((root / "data").glob("*.py")) + [root / "tasks" / "drugclip.py"]
    return {
        "schema": 1,
        "input_sha256": sha256_file(data_file),
        "checkpoints": [sha256_file(p) for p in checkpoints],
        "dictionaries": [sha256_file(p) for p in dictionaries],
        "settings": settings,
        "preprocessing": {str(p.relative_to(root)): sha256_file(p) for p in sources},
    }


def completed_prefix(mask, source):
    mask = np.asarray(mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError(f"{source}: expected a one-dimensional boolean completion mask")
    count = int(mask.sum())
    if not mask[:count].all() or mask[count:].any():
        raise ValueError(f"{source}: completion mask has gaps; use a new output directory")
    return count


def check_resume_file(handle, manifest, rows, folds=6, width=128):
    """Validate all folds before allowing any data or completion flag writes."""
    if rows <= 0:
        raise ValueError("Cannot encode an empty molecule library")
    expected = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    stored = handle.attrs.get("encoding_manifest")
    if stored is None:
        if len(handle):
            raise ValueError("Existing HDF5 has no encoding manifest; use a new output directory")
        handle.attrs["encoding_manifest"] = expected
        handle.flush()
    elif stored != expected:
        raise ValueError("HDF5 input, weights or encoding settings changed; use a new output directory")
    if "mol_reps" in handle:
        dset = handle["mol_reps"]
        if dset.shape != (rows, folds * width) or dset.dtype != np.float32:
            raise ValueError("HDF5 embedding shape or dtype does not match this run")
    for fold in range(folds):
        key = f"fold{fold}"
        if key in handle:
            mask = handle[key][:]
            if mask.shape != (rows,):
                raise ValueError(f"{key}: completion mask has the wrong row count")
            count = completed_prefix(mask, key)
            if count and "mol_reps" not in handle:
                raise ValueError(f"{key}: completion flags exist without embeddings")
