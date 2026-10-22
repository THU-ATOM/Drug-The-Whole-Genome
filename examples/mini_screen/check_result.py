"""Check mini-screen output counts, identifiers and finite descending scores."""
import argparse
import csv
import json
import math
from pathlib import Path


def check_result(manifest_path, result_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    with Path(result_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != manifest["expected_result_rows"]:
        raise ValueError(f"Expected {manifest['expected_result_rows']} output rows, got {len(rows)}")
    previous = math.inf
    seen = set()
    for row in rows:
        if len(row) != 2:
            raise ValueError(f"Expected smiles,score: {row!r}")
        smiles, score = row[0], float(row[1])
        if smiles not in manifest["smiles"] or smiles in seen:
            raise ValueError(f"Unexpected or repeated molecule: {smiles}")
        if not math.isfinite(score) or score > previous:
            raise ValueError("Scores must be finite and sorted in descending order")
        previous = score
        seen.add(smiles)
    print(f"Result checks passed: {len(rows)} row(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    check_result(args.manifest, args.result)
