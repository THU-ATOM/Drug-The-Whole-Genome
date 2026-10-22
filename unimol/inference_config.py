"""Shared inference configuration; importing this module does not load a model."""
import argparse
import math
from pathlib import Path

FOLD_VERSIONS = ("6_folds", "8_folds", "6_folds_filtered")


def str2bool(value):
    """Parse explicit CLI booleans, including the string 'False'."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1"):
        return True
    if value in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def fold_paths(version):
    """Return ordered checkpoint/cache paths relative to the repository root."""
    if version not in FOLD_VERSIONS:
        raise ValueError(f"Unknown fold version {version!r}; choose from {FOLD_VERSIONS}")
    count = 8 if version == "8_folds" else 6
    weights = "8_folds" if count == 8 else "6_folds"
    return (
        [f"./data/model_weights/{weights}/fold_{i}.pt" for i in range(count)],
        [f"./data/encoded_mol_embs/{version}/fold{i}.pkl" for i in range(count)],
    )


def validate_retrieval_inputs(pocket_path, mol_path, version, use_cache, save_path):
    """Fail before model allocation when required input files are absent."""
    checkpoints, caches = fold_paths(version)
    required = [("pocket LMDB", pocket_path)]
    required += [("checkpoint", path) for path in checkpoints]
    if use_cache:
        required += [("molecule cache", path) for path in caches]
    else:
        required.append(("molecule LMDB", mol_path))
    missing = [f"{kind}: {path or '<empty>'}" for kind, path in required
               if not path or not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))
    if not save_path:
        raise ValueError("--save-path must name an output file")
    output = Path(save_path).resolve()
    if any(output == Path(path).resolve() for _, path in required):
        raise ValueError("--save-path must not overwrite an input file")
    output.parent.mkdir(parents=True, exist_ok=True)


def write_ranked_results(scores, names, save_path):
    """Write the top 2%, preserving input order for tied scores.

    Nonempty libraries smaller than 50 molecules still produce one result.
    """
    if len(scores) != len(names):
        raise ValueError("Molecule names and scores must have the same length")
    if not len(scores):
        raise ValueError("Cannot rank an empty molecule library")
    if not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("Cannot rank NaN or infinite scores")
    ranked = sorted(zip(scores, names), key=lambda pair: pair[0], reverse=True)
    count = max(1, int(len(ranked) * 0.02))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        for score, name in ranked[:count]:
            handle.write(f"{name},{score}\n")
