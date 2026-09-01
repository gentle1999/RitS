#!/usr/bin/env python3
"""Measure intrinsic nearest-neighbor RMSD within each reference conformer set."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from megalodon.data.molecule_dataset import MoleculeDataset
from megalodon.interpolant.ot import rigid_alignment


def kabsch_rmsd(predicted, reference, mask=None):
    if mask is not None:
        predicted = predicted[mask]
        reference = reference[mask]
    aligned = rigid_alignment(predicted.float(), reference.float())
    return torch.sqrt(torch.mean(torch.sum((aligned - reference.float()) ** 2, dim=-1))).item()


def metric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
        return value[0] if value else ""
    return value


def reference_key(sample, index):
    reaction_id = str(scalar(getattr(sample, "reaction_id", "")) or "")
    if reaction_id:
        return f"reaction_id:{reaction_id}"
    dpa_ids = tuple(
        str(scalar(getattr(sample, name, "")) or "")
        for name in ("ene_id", "diene_id", "prod_id")
    )
    if any(dpa_ids):
        return "dpa:" + "_".join(dpa_ids)
    return f"sample:{index}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/dpa_ts"))
    parser.add_argument("--processed_folder", default="processed")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = MoleculeDataset(
        root=str(args.dataset_root),
        processed_folder=args.processed_folder,
        split=args.split,
    )
    groups = defaultdict(list)
    for index, sample in enumerate(dataset):
        groups[reference_key(sample, index)].append(
            {
                "dataset_index": index,
                "conf_id": str(scalar(getattr(sample, "conf_id", "")) or ""),
                "coords": sample.ts_coord.detach().cpu(),
                "numbers": sample.numbers.detach().cpu(),
            }
        )

    conformer_rows = []
    path_rows = []
    all_pairwise = []
    heavy_pairwise = []
    for key, conformers in groups.items():
        count = len(conformers)
        if count == 1:
            path_rows.append(
                {
                    "path": key,
                    "n_conformers": 1,
                    "mean_nearest_other_all_atom_rmsd": "",
                    "mean_nearest_other_heavy_atom_rmsd": "",
                }
            )
            continue

        all_matrix = np.full((count, count), np.inf, dtype=np.float64)
        heavy_matrix = np.full((count, count), np.inf, dtype=np.float64)
        for i in range(count):
            for j in range(i + 1, count):
                left = conformers[i]
                right = conformers[j]
                if not torch.equal(left["numbers"], right["numbers"]):
                    raise ValueError(f"Mapped atom order differs within {key}")
                heavy_mask = left["numbers"] > 1
                all_value = kabsch_rmsd(left["coords"], right["coords"])
                heavy_value = kabsch_rmsd(
                    left["coords"], right["coords"], mask=heavy_mask
                )
                all_matrix[i, j] = all_matrix[j, i] = all_value
                heavy_matrix[i, j] = heavy_matrix[j, i] = heavy_value
                all_pairwise.append(all_value)
                heavy_pairwise.append(heavy_value)

        nearest_all = all_matrix.min(axis=1)
        nearest_heavy = heavy_matrix.min(axis=1)
        for conformer, all_value, heavy_value in zip(
            conformers, nearest_all, nearest_heavy
        ):
            conformer_rows.append(
                {
                    "path": key,
                    "dataset_index": conformer["dataset_index"],
                    "conf_id": conformer["conf_id"],
                    "n_conformers": count,
                    "nearest_other_all_atom_rmsd": float(all_value),
                    "nearest_other_heavy_atom_rmsd": float(heavy_value),
                }
            )
        path_rows.append(
            {
                "path": key,
                "n_conformers": count,
                "mean_nearest_other_all_atom_rmsd": float(nearest_all.mean()),
                "mean_nearest_other_heavy_atom_rmsd": float(nearest_heavy.mean()),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    conformer_csv = args.output_dir / "per_conformer.csv"
    path_csv = args.output_dir / "per_path.csv"
    with conformer_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(conformer_rows[0]))
        writer.writeheader()
        writer.writerows(conformer_rows)
    with path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(path_rows[0]))
        writer.writeheader()
        writer.writerows(path_rows)

    multi_path_rows = [row for row in path_rows if row["n_conformers"] > 1]
    summary = {
        "dataset_root": str(args.dataset_root),
        "processed_folder": args.processed_folder,
        "split": args.split,
        "n_conformers": len(dataset),
        "n_paths": len(groups),
        "n_multiconformer_paths": len(multi_path_rows),
        "n_singleton_paths": len(groups) - len(multi_path_rows),
        "conformers_per_path": metric_summary([len(group) for group in groups.values()]),
        "definition": "For each reference, minimum Kabsch RMSD to another reference in the same path; self matches are excluded.",
        "atom_correspondence": "mapped atom order; no permutation matching",
        "nearest_other_reference_all_atom_rmsd_micro": metric_summary(
            [row["nearest_other_all_atom_rmsd"] for row in conformer_rows]
        ),
        "nearest_other_reference_heavy_atom_rmsd_micro": metric_summary(
            [row["nearest_other_heavy_atom_rmsd"] for row in conformer_rows]
        ),
        "path_mean_nearest_other_all_atom_rmsd_macro": metric_summary(
            [row["mean_nearest_other_all_atom_rmsd"] for row in multi_path_rows]
        ),
        "path_mean_nearest_other_heavy_atom_rmsd_macro": metric_summary(
            [row["mean_nearest_other_heavy_atom_rmsd"] for row in multi_path_rows]
        ),
        "all_reference_pairwise_all_atom_rmsd": metric_summary(all_pairwise),
        "all_reference_pairwise_heavy_atom_rmsd": metric_summary(heavy_pairwise),
        "per_conformer_csv": str(conformer_csv.resolve()),
        "per_path_csv": str(path_csv.resolve()),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
