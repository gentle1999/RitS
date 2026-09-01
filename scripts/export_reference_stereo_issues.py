#!/usr/bin/env python3
"""Package reference stereo violations with mapped reaction metadata and XYZ files."""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from megalodon.data.molecule_dataset import MoleculeDataset


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar(value):
    try:
        values = value.detach().cpu().reshape(-1).tolist()
        return values[0] if values else ""
    except AttributeError:
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_checks", type=Path, required=True)
    parser.add_argument("--constraint_details", type=Path, required=True)
    parser.add_argument("--reaction_csv", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/dpa_ts"))
    parser.add_argument("--processed_folder", default="processed")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--source_xyz_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    reference_rows = read_csv(args.reference_checks)
    detail_rows = [
        row
        for row in read_csv(args.constraint_details)
        if row["source"] == "reference" and row["status"] == "opposite"
    ]
    reaction_rows = read_csv(args.reaction_csv)
    reactions = {row["reaction_id"]: row for row in reaction_rows}
    dataset = MoleculeDataset(
        root=str(args.dataset_root),
        processed_folder=args.processed_folder,
        split=args.split,
    )

    samples = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        sample_id = str(scalar(getattr(sample, "id", "")))
        reaction_id = str(scalar(getattr(sample, "reaction_id", "")))
        source_xyz = args.source_xyz_dir / f"{sample_id}.xyz"
        if not source_xyz.exists():
            raise FileNotFoundError(source_xyz)
        samples[index] = {
            "dataset_index": index,
            "sample_id": sample_id,
            "reaction_id": reaction_id,
            "conf_id": str(scalar(getattr(sample, "conf_id", ""))),
            "ene_id": str(scalar(getattr(sample, "ene_id", ""))),
            "diene_id": str(scalar(getattr(sample, "diene_id", ""))),
            "prod_id": str(scalar(getattr(sample, "prod_id", ""))),
            "source_xyz": source_xyz,
        }

    details_by_index = defaultdict(list)
    for row in detail_rows:
        details_by_index[int(row["candidate_index"])].append(row)

    checks_by_path = defaultdict(list)
    for row in reference_rows:
        checks_by_path[row["path"]].append(row)
    issue_paths = {
        path
        for path, rows in checks_by_path.items()
        if any(int(row["pass_all_stereo"]) == 0 for row in rows)
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_path_dir = args.output_dir / "by_path"
    by_path_dir.mkdir(exist_ok=True)
    path_summary_rows = []
    reference_issue_rows = []
    enriched_detail_rows = []
    nested_report = []

    ordered_paths = sorted(
        issue_paths,
        key=lambda path: (
            any(int(row["pass_all_stereo"]) for row in checks_by_path[path]),
            path,
        ),
    )
    for path in ordered_paths:
        checks = sorted(
            checks_by_path[path], key=lambda row: int(row["dataset_index"])
        )
        sample = samples[int(checks[0]["dataset_index"])]
        reaction_id = sample["reaction_id"]
        reaction = reactions[reaction_id]
        path_dir = by_path_dir / reaction_id
        violating_dir = path_dir / "violating"
        passing_dir = path_dir / "passing"
        violating_dir.mkdir(parents=True, exist_ok=True)
        passing_dir.mkdir(parents=True, exist_ok=True)

        n_passing = sum(int(row["pass_all_stereo"]) for row in checks)
        n_failing = len(checks) - n_passing
        all_fail = n_passing == 0
        path_details = []
        conformer_report = []
        for check in checks:
            index = int(check["dataset_index"])
            sample_info = samples[index]
            passed = bool(int(check["pass_all_stereo"]))
            destination_dir = passing_dir if passed else violating_dir
            copied_xyz = destination_dir / sample_info["source_xyz"].name
            shutil.copy2(sample_info["source_xyz"], copied_xyz)
            violations = details_by_index[index]
            if not passed:
                reference_issue_rows.append(
                    {
                        "priority": "all_references_fail" if all_fail else "partial_failure",
                        "path": path,
                        "reaction_id": reaction_id,
                        "ene_id": sample_info["ene_id"],
                        "diene_id": sample_info["diene_id"],
                        "prod_id": sample_info["prod_id"],
                        "dataset_index": index,
                        "sample_id": sample_info["sample_id"],
                        "conf_id": sample_info["conf_id"],
                        "n_violations": len(violations),
                        "violations": ";".join(
                            violation["constraint"] for violation in violations
                        ),
                        "source_xyz": str(sample_info["source_xyz"].resolve()),
                        "packaged_xyz": str(copied_xyz.resolve()),
                        "rxn_smiles": reaction["rxn_smiles"],
                    }
                )
            for violation in violations:
                enriched = {
                    "priority": "all_references_fail" if all_fail else "partial_failure",
                    "reaction_id": reaction_id,
                    "ene_id": sample_info["ene_id"],
                    "diene_id": sample_info["diene_id"],
                    "prod_id": sample_info["prod_id"],
                    "dataset_index": index,
                    "sample_id": sample_info["sample_id"],
                    "conf_id": sample_info["conf_id"],
                    "source_xyz": str(sample_info["source_xyz"].resolve()),
                    "packaged_xyz": str(copied_xyz.resolve()),
                    "rxn_smiles": reaction["rxn_smiles"],
                    **violation,
                }
                enriched_detail_rows.append(enriched)
                path_details.append(enriched)
            conformer_report.append(
                {
                    "dataset_index": index,
                    "sample_id": sample_info["sample_id"],
                    "conf_id": sample_info["conf_id"],
                    "pass": passed,
                    "xyz": str(copied_xyz.resolve()),
                    "violations": [
                        {
                            "constraint": violation["constraint"],
                            "side": violation["side"],
                            "kind": violation["kind"],
                            "descriptor": violation["descriptor"],
                            "value": float(violation["value"]),
                            "signed_margin": float(violation["signed_margin"]),
                        }
                        for violation in violations
                    ],
                }
            )

        unique_constraints = sorted(
            {detail["constraint"] for detail in path_details}
        )
        types = sorted(
            {f"{detail['side']}:{detail['kind']}" for detail in path_details}
        )
        reaction_text = [
            f"reaction_id: {reaction_id}",
            f"ene_id: {sample['ene_id']}",
            f"diene_id: {sample['diene_id']}",
            f"prod_id: {sample['prod_id']}",
            f"all_references_fail: {all_fail}",
            f"n_references: {len(checks)}",
            f"n_passing: {n_passing}",
            f"n_failing: {n_failing}",
            "",
            "mapped rxn_smiles:",
            reaction["rxn_smiles"],
            "",
            "violated constraints:",
            *unique_constraints,
            "",
        ]
        (path_dir / "reaction.txt").write_text("\n".join(reaction_text))
        path_summary_rows.append(
            {
                "priority": "all_references_fail" if all_fail else "partial_failure",
                "path": path,
                "reaction_id": reaction_id,
                "ene_id": sample["ene_id"],
                "diene_id": sample["diene_id"],
                "prod_id": sample["prod_id"],
                "n_references": len(checks),
                "n_passing": n_passing,
                "n_failing": n_failing,
                "failure_fraction": n_failing / len(checks),
                "violation_types": ";".join(types),
                "violated_constraints": ";".join(unique_constraints),
                "path_directory": str(path_dir.resolve()),
                "rxn_smiles": reaction["rxn_smiles"],
            }
        )
        nested_report.append(
            {
                "priority": "all_references_fail" if all_fail else "partial_failure",
                "path": path,
                "reaction_id": reaction_id,
                "ene_id": sample["ene_id"],
                "diene_id": sample["diene_id"],
                "prod_id": sample["prod_id"],
                "rxn_smiles": reaction["rxn_smiles"],
                "n_references": len(checks),
                "n_passing": n_passing,
                "n_failing": n_failing,
                "violated_constraints": unique_constraints,
                "conformers": conformer_report,
            }
        )

    path_summary_csv = args.output_dir / "path_summary.csv"
    problematic_csv = args.output_dir / "problematic_references.csv"
    violation_csv = args.output_dir / "violations.csv"
    write_csv(path_summary_csv, path_summary_rows)
    write_csv(problematic_csv, reference_issue_rows)
    write_csv(violation_csv, enriched_detail_rows)
    report_json = args.output_dir / "report.json"
    report_json.write_text(json.dumps(nested_report, indent=2, ensure_ascii=True) + "\n")

    all_fail_rows = [row for row in path_summary_rows if row["priority"] == "all_references_fail"]
    partial_rows = [row for row in path_summary_rows if row["priority"] == "partial_failure"]
    readme_lines = [
        "# Reference stereo issues",
        "",
        "Only opposite-sign stereo constraints are treated as violations. Near-zero values are allowed.",
        "",
        f"- Problematic reference structures: {len(reference_issue_rows)}",
        f"- Paths with issues: {len(path_summary_rows)}",
        f"- Paths where all references fail: {len(all_fail_rows)}",
        f"- Paths with mixed passing/failing references: {len(partial_rows)}",
        "",
        "Each `by_path/<reaction_id>/reaction.txt` contains the full mapped reaction SMILES.",
        "The `violating/` and `passing/` subdirectories provide direct geometry comparisons.",
        "",
        "## All references fail",
        "",
    ]
    readme_lines.extend(
        f"- `{row['reaction_id']}`: {row['n_failing']}/{row['n_references']} fail; "
        f"`by_path/{row['reaction_id']}/reaction.txt`"
        for row in all_fail_rows
    )
    readme_lines.extend(["", "## Partial failures", ""])
    readme_lines.extend(
        f"- `{row['reaction_id']}`: {row['n_failing']}/{row['n_references']} fail; "
        f"`by_path/{row['reaction_id']}/reaction.txt`"
        for row in partial_rows
    )
    (args.output_dir / "README.md").write_text("\n".join(readme_lines) + "\n")

    summary = {
        "n_problematic_reference_structures": len(reference_issue_rows),
        "n_paths_with_issues": len(path_summary_rows),
        "n_all_reference_fail_paths": len(all_fail_rows),
        "n_partial_failure_paths": len(partial_rows),
        "path_summary_csv": str(path_summary_csv.resolve()),
        "problematic_references_csv": str(problematic_csv.resolve()),
        "violations_csv": str(violation_csv.resolve()),
        "report_json": str(report_json.resolve()),
        "by_path_dir": str(by_path_dir.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
