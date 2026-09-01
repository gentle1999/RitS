#!/usr/bin/env python3
"""Report how RMSD clustering thresholds change per-path candidate counts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_ts_xtb_ranking import cluster_candidates, pairwise_kabsch_rmsd


def metric(values: list[int]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n_paths": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": int(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": int(array.max()),
        "fraction_one_cluster": float(np.mean(array == 1)),
        "fraction_le_5_clusters": float(np.mean(array <= 5)),
        "fraction_le_10_clusters": float(np.mean(array <= 10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    rows = []
    summary = {"cache": str(args.cache.resolve()), "thresholds": args.thresholds, "by_atoms": {}}
    for atom_mode in ("all", "heavy"):
        summary["by_atoms"][atom_mode] = {}
        counts_by_threshold = {threshold: [] for threshold in args.thresholds}
        for record in cache["records"]:
            mask = record["numbers"] > 1 if atom_mode == "heavy" else record["numbers"] > 0
            distances = pairwise_kabsch_rmsd(record["candidates"], mask)
            for threshold in args.thresholds:
                labels = cluster_candidates(distances, threshold)
                counts_by_threshold[threshold].append(int(labels.max()))
        for threshold in args.thresholds:
            stats = metric(counts_by_threshold[threshold])
            summary["by_atoms"][atom_mode][str(threshold)] = stats
            rows.append({"atom_mode": atom_mode, "threshold_angstrom": threshold, **stats})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
