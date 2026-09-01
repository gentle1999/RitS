#!/usr/bin/env python3
"""Audit every raw DPA TS conformer against mapped reaction stereochemistry."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from check_ts_stereo_constraints import (
    evaluate_coords,
    extract_reaction_constraints,
    load_reactions,
    read_xyz,
)


DPA_XYZ_NAME_RE = re.compile(
    r"^(?P<ene>\d+)_(?P<diene>\d+)_(?P<prod>\d+)_conf_(?P<conf>\d+)"
    r"_ts\.(?P<digest>[0-9a-fA-F]+)\.xyz$"
)


def normalized_id(value: str) -> str:
    return str(int(str(value).strip()))


def parse_filename(path: Path) -> tuple[tuple[str, str, str], str]:
    match = DPA_XYZ_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unsupported DPA XYZ filename: {path.name}")
    key = tuple(normalized_id(match.group(name)) for name in ("ene", "diene", "prod"))
    return key, match.group("conf")


def read_split_metadata(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["file"]: row["split"] for row in rows}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-csv", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-metadata",
        type=Path,
        help="Optional existing metadata.csv used only to annotate the unchanged split.",
    )
    args = parser.parse_args()

    _, reactions = load_reactions(args.reaction_csv)
    split_by_file = read_split_metadata(args.split_metadata)
    xyz_paths = sorted(args.xyz_dir.glob("*.xyz"))
    if not xyz_paths:
        raise ValueError(f"no XYZ files found in {args.xyz_dir}")

    constraint_cache = {}
    check_rows = []
    violation_rows = []
    malformed_files = []
    missing_reaction_files = []
    paths_with_input = defaultdict(int)
    paths_with_pass = defaultdict(int)

    for xyz_path in tqdm(xyz_paths, desc="Auditing all DPA TS conformers"):
        try:
            key, conf_id = parse_filename(xyz_path)
        except ValueError:
            malformed_files.append(xyz_path.name)
            continue
        if key not in reactions:
            missing_reaction_files.append(xyz_path.name)
            continue

        reaction = reactions[key]
        reaction_id = reaction["reaction_id"].strip()
        if reaction_id not in constraint_cache:
            constraint_cache[reaction_id] = extract_reaction_constraints(
                reaction["rxn_smiles"]
            )
        constraints, skipped, expected_numbers = constraint_cache[reaction_id]
        numbers, coords = read_xyz(xyz_path)
        mapping_ok = bool(np.array_equal(numbers, expected_numbers))
        stereo_pass, details = evaluate_coords(coords, constraints)
        passed = bool(mapping_ok and stereo_pass)
        violations = [detail["constraint"] for detail in details if not detail["pass"]]
        split = split_by_file.get(xyz_path.name, "")
        path_key = "_".join(key)
        paths_with_input[path_key] += 1
        paths_with_pass[path_key] += int(passed)

        row = {
            "file": xyz_path.name,
            "sample_id": xyz_path.stem,
            "source_xyz": str(xyz_path.resolve()),
            "ene_id": key[0],
            "diene_id": key[1],
            "prod_id": key[2],
            "conf_id": conf_id,
            "reaction_id": reaction_id,
            "split": split,
            "mapping_order_valid": int(mapping_ok),
            "n_constraints": len(constraints),
            "n_tetrahedral_constraints": sum(
                constraint.kind == "tetrahedral" for constraint in constraints
            ),
            "n_double_bond_constraints": sum(
                constraint.kind == "double_bond" for constraint in constraints
            ),
            "n_skipped_n_constraints": len(skipped),
            "n_violations": len(violations) + int(not mapping_ok),
            "pass_all_stereo": int(passed),
            "violations": ";".join(violations),
            "reason": (
                "mapped atom order mismatch"
                if not mapping_ok
                else "non-N stereo sign violation"
                if violations
                else ""
            ),
            "rxn_smiles": reaction["rxn_smiles"],
        }
        check_rows.append(row)
        if not passed:
            violation_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(check_rows[0])
    write_csv(args.output_dir / "all_reference_stereo_checks.csv", check_rows, fields)
    write_csv(args.output_dir / "excluded_references.csv", violation_rows, fields)

    invalid_paths = {
        "_".join(row[name] for name in ("ene_id", "diene_id", "prod_id"))
        for row in violation_rows
    }
    exhausted_paths = sorted(path for path in invalid_paths if paths_with_pass[path] == 0)
    constraint_counts = Counter()
    skipped_count = 0
    for constraints, skipped, _ in constraint_cache.values():
        skipped_count += len(skipped)
        constraint_counts.update(
            f"{constraint.side}_{constraint.kind}" for constraint in constraints
        )
    split_total = Counter(row["split"] or "unassigned" for row in check_rows)
    split_invalid = Counter(row["split"] or "unassigned" for row in violation_rows)
    summary = {
        "reaction_csv": str(args.reaction_csv.resolve()),
        "xyz_dir": str(args.xyz_dir.resolve()),
        "split_metadata": str(args.split_metadata.resolve()) if args.split_metadata else None,
        "policy": {
            "atom_index": "atom_map - 1; maps must be contiguous 1..N",
            "rejection": "reject only when expected_sign * value < 0",
            "zero_policy": "zero, near-zero, and undefined values are accepted",
            "nitrogen_policy": (
                "ignore tetrahedral centers on N and E/Z bonds with N at either endpoint; "
                "N substituents on non-N centers remain constrained"
            ),
        },
        "xyz_files_found": len(xyz_paths),
        "checked_structures": len(check_rows),
        "passing_structures": sum(int(row["pass_all_stereo"]) for row in check_rows),
        "excluded_structures": len(violation_rows),
        "pass_fraction": float(np.mean([int(row["pass_all_stereo"]) for row in check_rows])),
        "paths_checked": len(paths_with_input),
        "paths_with_exclusions": len(invalid_paths),
        "paths_with_no_passing_conformer": len(exhausted_paths),
        "path_keys_with_no_passing_conformer": exhausted_paths,
        "by_split": {
            split: {
                "checked": split_total[split],
                "excluded": split_invalid[split],
                "passing": split_total[split] - split_invalid[split],
            }
            for split in sorted(split_total)
        },
        "mapping_order_failures": sum(not int(row["mapping_order_valid"]) for row in check_rows),
        "malformed_xyz_files": malformed_files,
        "missing_reaction_files": missing_reaction_files,
        "reactions_with_constraints_cached": len(constraint_cache),
        "constraint_definitions_by_kind_and_side": dict(constraint_counts),
        "skipped_n_constraint_definitions": skipped_count,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
