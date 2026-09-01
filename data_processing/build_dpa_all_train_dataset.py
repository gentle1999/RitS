#!/usr/bin/env python3
"""Build a full-DPA training view with a configurable diagnostic split."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import torch
from torch_geometric.data.collate import collate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from megalodon.data.molecule_dataset import MoleculeDataset


def save_dataset(items, path: Path) -> None:
    if not items:
        raise ValueError(f"cannot save empty dataset: {path}")
    batch = collate(items[0].__class__, items, increment=False, add_batch=False)
    torch.save(batch[:2], path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpa-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument(
        "--validation-label",
        default="validation diagnostic split (provenance unspecified)",
        help="Human-readable validation provenance stored in summary.json.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--processed-folder", default="processed")
    args = parser.parse_args()

    dpa_datasets = {
        split: MoleculeDataset(
            root=str(args.dpa_root),
            processed_folder=args.processed_folder,
            split=split,
        )
        for split in ("train", "val", "test")
    }
    validation = MoleculeDataset(
        root=str(args.validation_root),
        processed_folder=args.processed_folder,
        split="val",
    )
    all_dpa = [sample for split in ("train", "val", "test") for sample in dpa_datasets[split]]

    processed_dir = args.output_root / args.processed_folder
    processed_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(all_dpa, processed_dir / "train_h.pt")
    shutil.copy2(
        args.validation_root / args.processed_folder / "val_h.pt",
        processed_dir / "val_h.pt",
    )
    # Retain the original DPA test tensor for fitting diagnostics. It is part of
    # train_h.pt in this view and therefore is not an independent test set.
    shutil.copy2(
        args.dpa_root / args.processed_folder / "test_h.pt",
        processed_dir / "test_h.pt",
    )

    source_metadata = args.dpa_root / "metadata.csv"
    if source_metadata.exists():
        with source_metadata.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0]) + ["included_in_full_training"]
        with (args.output_root / "metadata.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "included_in_full_training": 1})

    summary = {
        "dpa_root": str(args.dpa_root.resolve()),
        "validation_root": str(args.validation_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "training_source_splits": ["train", "val", "test"],
        "dpa_source_split_files": {
            split: len(dataset) for split, dataset in dpa_datasets.items()
        },
        "train_files": len(all_dpa),
        "validation_files": len(validation),
        "validation_definition": args.validation_label,
        "test_files": len(dpa_datasets["test"]),
        "test_definition": (
            "original DPA test split, also included in train_h.pt; fitting diagnostic only"
        ),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
