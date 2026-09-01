#!/usr/bin/env python3
"""Select the lowest-xTB candidate from every RMSD cluster within an energy window."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--energy-threshold", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.xyz_manifest)
    if not rows:
        raise ValueError(f"empty XYZ manifest: {args.xyz_manifest}")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not row.get("xtb_energy_hartree") or row["xtb_energy_hartree"] == "nan":
            continue
        grouped[row["path"]].append(row)

    selected_dir = args.output_dir / "xyz"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = []
    path_rows = []
    for path, path_rows_all in sorted(
        grouped.items(), key=lambda item: int(item[1][0]["global_path_index"])
    ):
        energies = [float(row["xtb_energy_hartree"]) for row in path_rows_all]
        minimum = min(energies)
        kept = [
            row for row in path_rows_all
            if float(row["xtb_energy_hartree"]) - minimum <= args.energy_threshold + 1e-12
        ]
        for row in kept:
            source = Path(row["xyz_path"])
            destination = selected_dir / source.name
            shutil.copy2(source, destination)
            selected = dict(row)
            selected["path_energy_min_hartree"] = f"{minimum:.12f}"
            selected["energy_delta_hartree"] = f"{float(row['xtb_energy_hartree']) - minimum:.12f}"
            selected["energy_threshold_hartree"] = f"{args.energy_threshold:.12f}"
            selected["xyz_path"] = str(destination.resolve())
            selected_rows.append(selected)
        path_rows.append(
            {
                "path": path,
                "global_path_index": path_rows_all[0]["global_path_index"],
                "n_clusters_with_energy": len(path_rows_all),
                "path_energy_min_hartree": f"{minimum:.12f}",
                "energy_threshold_hartree": f"{args.energy_threshold:.12f}",
                "n_selected_clusters": len(kept),
                "n_removed_clusters": len(path_rows_all) - len(kept),
            }
        )

    manifest = args.output_dir / "selected_xyz_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    path_csv = args.output_dir / "selected_path_summary.csv"
    with path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(path_rows[0]))
        writer.writeheader()
        writer.writerows(path_rows)

    source_summary = json.loads(args.summary.read_text())
    summary = {
        "source_manifest": str(args.xyz_manifest.resolve()),
        "source_summary": str(args.summary.resolve()),
        "cluster_method": source_summary["cluster_method"],
        "cluster_threshold_angstrom": source_summary["cluster_threshold_angstrom"],
        "energy_method": source_summary["xtb"]["method"],
        "energy_threshold_hartree": args.energy_threshold,
        "energy_threshold_kcal_per_mol": args.energy_threshold * 627.509474,
        "n_paths_with_energy": len(path_rows),
        "n_source_cluster_representatives": len(rows),
        "n_selected_cluster_representatives": len(selected_rows),
        "selected_per_path": {
            "mean": sum(int(row["n_selected_clusters"]) for row in path_rows) / len(path_rows),
            "min": min(int(row["n_selected_clusters"]) for row in path_rows),
            "median": sorted(int(row["n_selected_clusters"]) for row in path_rows)[len(path_rows) // 2],
            "max": max(int(row["n_selected_clusters"]) for row in path_rows),
        },
        "outputs": {
            "selected_xyz_dir": str(selected_dir.resolve()),
            "selected_manifest": str(manifest.resolve()),
            "selected_path_summary": str(path_csv.resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
