"""Target selection and per-run benchmark reports."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np

RATIOS = ("0.005", "0.01", "0.02", "0.05")
METRICS = ("auc", "bedroc") + tuple(f"{kind}_{ratio}" for kind in ("ef", "re") for ratio in RATIOS)


def select_targets(root, requested=None):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Benchmark directory not found: {root}")
    available = sorted(p.name for p in root.iterdir() if p.is_dir())
    selected = sorted(set(requested)) if requested else available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown targets in {root}: {', '.join(unknown)}")
    if not selected:
        raise ValueError(f"No target directories in {root}")
    return selected


def run_benchmark(benchmark, evaluate, use_folds, results_dir="./results/benchmarks",
                  targets=None, metadata=None):
    root = Path("data/DUD-E" if benchmark == "DUDE" else "data/lit_pcba")
    selected = select_targets(root, targets)
    pocket_file = "pocket.lmdb" if benchmark == "DUDE" else "pockets.lmdb"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = Path(results_dir) / f"{benchmark.lower()}-{stamp}-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    records = []

    def save():
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("target", "status", "reason") + METRICS)
            writer.writeheader()
            writer.writerows(records)
        valid = [row for row in records if row["status"] == "completed"]
        report = {
            "benchmark": benchmark, "use_folds": bool(use_folds),
            "started_at_utc": stamp, "requested_targets": selected,
            "counts": {status: sum(r["status"] == status for r in records)
                       for status in ("completed", "skipped", "failed")},
            "means": {key: float(np.mean([r[key] for r in valid])) for key in METRICS} if valid else {},
            "records": records, "run": metadata or {},
        }
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    for target in selected:
        row = {"target": target, "status": "skipped", "reason": ""}
        missing = [name for name in ("mols.lmdb", pocket_file) if not (root / target / name).is_file()]
        if missing:
            row["reason"] = "Missing " + ", ".join(missing)
            records.append(row)
            save()
            continue
        try:
            result = evaluate(target)
            if result is None:
                row["reason"] = "Evaluator returned no result"
            else:
                auc, bedroc, ef, re = result[:4]
                values = {"auc": float(auc), "bedroc": float(bedroc)}
                values.update({f"{kind}_{ratio}": float(data[ratio])
                               for kind, data in (("ef", ef), ("re", re)) for ratio in RATIOS})
                if not all(np.isfinite(v) for v in values.values()):
                    raise ValueError("Benchmark metrics contain NaN or infinity")
                row.update(values, status="completed")
        except Exception as exc:
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            records.append(row)
            save()
            raise RuntimeError(f"{benchmark}/{target} failed; report: {output}") from exc
        records.append(row)
        save()
    print(f"Benchmark report: {output}")
    if not any(row["status"] == "completed" for row in records):
        raise ValueError(f"No {benchmark} targets were evaluated; see {output}")
    return output
