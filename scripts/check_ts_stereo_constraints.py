#!/usr/bin/env python3
"""Reject generated TS coordinates that violate mapped stereo constraints."""

import argparse
import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from megalodon.data.molecule_dataset import MoleculeDataset


@dataclass(frozen=True)
class StereoConstraint:
    kind: str
    side: str
    anchor: str
    atom_maps: tuple[int, ...]
    expected_sign: int
    descriptor: str

    @property
    def identifier(self):
        maps = "-".join(map(str, self.atom_maps))
        return f"{self.side}:{self.kind}:{self.anchor}:{maps}:{self.descriptor}"


def normalized_id(value):
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def load_reactions(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"reaction_id", "ene_id", "diene_id", "prod_id", "rxn_smiles"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"Reaction CSV missing columns: {sorted(missing)}")
    by_reaction_id = {}
    by_dpa_key = {}
    for row in rows:
        reaction_id = row["reaction_id"].strip()
        if reaction_id:
            by_reaction_id[reaction_id] = row
        key = tuple(normalized_id(row[name]) for name in ("ene_id", "diene_id", "prod_id"))
        by_dpa_key[key] = row
    return by_reaction_id, by_dpa_key


def parse_mapped_side(smiles):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError(f"RDKit failed to parse mapped SMILES: {smiles}")
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if any(atom_map <= 0 for atom_map in maps) or len(set(maps)) != len(maps):
        raise ValueError("Mapped SMILES has missing or duplicate atom maps")
    return mol


def cip_rank(atom):
    if not atom.HasProp("_CIPRank"):
        raise ValueError(f"Atom map {atom.GetAtomMapNum()} has no CIP rank")
    return atom.GetIntProp("_CIPRank")


def extract_side_constraints(mol, side):
    constraints = []
    skipped = []
    for atom in mol.GetAtoms():
        if not atom.HasProp("_CIPCode"):
            continue
        descriptor = atom.GetProp("_CIPCode")
        if descriptor not in {"R", "S"}:
            continue
        if atom.GetAtomicNum() == 7:
            skipped.append(
                {
                    "side": side,
                    "kind": "tetrahedral",
                    "anchor": str(atom.GetAtomMapNum()),
                    "reason": "nitrogen-centered stereo is allowed to invert",
                }
            )
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 4:
            skipped.append(
                {
                    "side": side,
                    "kind": "tetrahedral",
                    "anchor": str(atom.GetAtomMapNum()),
                    "reason": f"expected 4 explicit neighbors, found {len(neighbors)}",
                }
            )
            continue
        ordered = sorted(neighbors, key=cip_rank, reverse=True)
        ranks = [cip_rank(neighbor) for neighbor in ordered]
        if len(set(ranks)) != 4:
            skipped.append(
                {
                    "side": side,
                    "kind": "tetrahedral",
                    "anchor": str(atom.GetAtomMapNum()),
                    "reason": "non-unique CIP neighbor ranks",
                }
            )
            continue
        constraints.append(
            StereoConstraint(
                kind="tetrahedral",
                side=side,
                anchor=str(atom.GetAtomMapNum()),
                atom_maps=tuple(neighbor.GetAtomMapNum() for neighbor in ordered),
                # For det([p1-p4, p2-p4, p3-p4]) with descending CIP priority.
                expected_sign=-1 if descriptor == "R" else 1,
                descriptor=descriptor,
            )
        )

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo not in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}:
            continue
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetAtomicNum() == 7 or end.GetAtomicNum() == 7:
            skipped.append(
                {
                    "side": side,
                    "kind": "double_bond",
                    "anchor": f"{begin.GetAtomMapNum()}-{end.GetAtomMapNum()}",
                    "reason": "nitrogen-centered E/Z stereo is allowed to invert",
                }
            )
            continue
        stereo_atoms = list(bond.GetStereoAtoms())
        if len(stereo_atoms) != 2:
            skipped.append(
                {
                    "side": side,
                    "kind": "double_bond",
                    "anchor": f"{begin.GetAtomMapNum()}-{end.GetAtomMapNum()}",
                    "reason": f"expected 2 RDKit stereo atoms, found {len(stereo_atoms)}",
                }
            )
            continue
        first = mol.GetAtomWithIdx(stereo_atoms[0])
        second = mol.GetAtomWithIdx(stereo_atoms[1])
        begin_neighbor_indices = {atom.GetIdx() for atom in begin.GetNeighbors()}
        end_neighbor_indices = {atom.GetIdx() for atom in end.GetNeighbors()}
        if first.GetIdx() in begin_neighbor_indices and second.GetIdx() in end_neighbor_indices:
            begin_stereo_atom, end_stereo_atom = first, second
        elif second.GetIdx() in begin_neighbor_indices and first.GetIdx() in end_neighbor_indices:
            begin_stereo_atom, end_stereo_atom = second, first
        else:
            skipped.append(
                {
                    "side": side,
                    "kind": "double_bond",
                    "anchor": f"{begin.GetAtomMapNum()}-{end.GetAtomMapNum()}",
                    "reason": "RDKit stereo atoms do not bracket the double bond",
                }
            )
            continue
        descriptor = "E" if stereo == Chem.BondStereo.STEREOE else "Z"
        constraints.append(
            StereoConstraint(
                kind="double_bond",
                side=side,
                anchor=f"{begin.GetAtomMapNum()}-{end.GetAtomMapNum()}",
                atom_maps=(
                    begin_stereo_atom.GetAtomMapNum(),
                    begin.GetAtomMapNum(),
                    end.GetAtomMapNum(),
                    end_stereo_atom.GetAtomMapNum(),
                ),
                expected_sign=-1 if descriptor == "E" else 1,
                descriptor=descriptor,
            )
        )
    return constraints, skipped


def extract_reaction_constraints(reaction_smiles):
    reactants, products = reaction_smiles.split(">>")
    r_mol = parse_mapped_side(reactants)
    p_mol = parse_mapped_side(products)
    r_constraints, r_skipped = extract_side_constraints(r_mol, "reactant")
    p_constraints, p_skipped = extract_side_constraints(p_mol, "product")
    numbers_by_map = {
        atom.GetAtomMapNum(): atom.GetAtomicNum() for atom in r_mol.GetAtoms()
    }
    product_numbers = {
        atom.GetAtomMapNum(): atom.GetAtomicNum() for atom in p_mol.GetAtoms()
    }
    if numbers_by_map != product_numbers:
        raise ValueError("Reactant/product elements differ by atom map")
    expected_maps = list(range(1, len(numbers_by_map) + 1))
    if sorted(numbers_by_map) != expected_maps:
        raise ValueError("Atom maps are not contiguous 1..N")
    constraints = r_constraints + p_constraints
    return constraints, r_skipped + p_skipped, np.asarray(
        [numbers_by_map[atom_map] for atom_map in expected_maps], dtype=np.int64
    )


def tetrahedral_volume(coords, atom_maps):
    points = coords[np.asarray(atom_maps, dtype=np.int64) - 1]
    vectors = np.stack(
        [points[0] - points[3], points[1] - points[3], points[2] - points[3]]
    )
    denominator = np.prod(np.linalg.norm(vectors, axis=1))
    if denominator <= 1e-12:
        return np.nan
    return float(np.linalg.det(vectors) / denominator)


def double_bond_order(coords, atom_maps):
    left, begin, end, right = coords[np.asarray(atom_maps, dtype=np.int64) - 1]
    axis = end - begin
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 1e-12:
        return np.nan
    axis /= axis_norm
    left_vector = left - begin
    right_vector = right - end
    left_projected = left_vector - np.dot(left_vector, axis) * axis
    right_projected = right_vector - np.dot(right_vector, axis) * axis
    denominator = np.linalg.norm(left_projected) * np.linalg.norm(right_projected)
    if denominator <= 1e-12:
        return np.nan
    return float(np.dot(left_projected, right_projected) / denominator)


def evaluate_coords(coords, constraints):
    details = []
    for constraint in constraints:
        if constraint.kind == "tetrahedral":
            value = tetrahedral_volume(coords, constraint.atom_maps)
        else:
            value = double_bond_order(coords, constraint.atom_maps)
        signed_margin = constraint.expected_sign * value if np.isfinite(value) else np.nan
        opposite = bool(np.isfinite(signed_margin) and signed_margin < 0.0)
        status = (
            "undefined"
            if not np.isfinite(signed_margin)
            else "zero"
            if signed_margin == 0.0
            else "opposite"
            if opposite
            else "same_sign"
        )
        details.append(
            {
                "constraint": constraint.identifier,
                "kind": constraint.kind,
                "side": constraint.side,
                "descriptor": constraint.descriptor,
                "value": value,
                "signed_margin": signed_margin,
                "status": status,
                "opposite_sign": int(opposite),
                "pass": not opposite,
            }
        )
    return all(detail["pass"] for detail in details), details


def read_xyz(path):
    lines = path.read_text().splitlines()
    atom_count = int(lines[0].strip())
    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError(f"Truncated XYZ: {path}")
    periodic_table = Chem.GetPeriodicTable()
    numbers = []
    coords = []
    for line in atom_lines:
        fields = line.split()
        numbers.append(periodic_table.GetAtomicNumber(fields[0]))
        coords.append([float(value) for value in fields[1:4]])
    return np.asarray(numbers, dtype=np.int64), np.asarray(coords, dtype=np.float64)


def reaction_for_path(path_key, by_reaction_id, by_dpa_key):
    if path_key.startswith("reaction_id:"):
        return by_reaction_id[path_key.split(":", 1)[1]]
    if path_key.startswith("dpa:"):
        return by_dpa_key[tuple(path_key.split(":", 1)[1].split("_"))]
    raise KeyError(f"Unsupported path key: {path_key}")


def metric_summary(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def path_topk_summary(rows, top_k):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["path"]].append(row)
    summary = {}
    for k in top_k:
        all_pass = []
        constrained_pass = []
        for path_rows in grouped.values():
            eligible = [row for row in path_rows if int(row["rank"]) <= k]
            passed = any(int(row["pass_all_stereo"]) for row in eligible)
            all_pass.append(passed)
            if int(path_rows[0]["n_constraints"]) > 0:
                constrained_pass.append(passed)
        summary[f"top{k}"] = {
            "all_paths_any_passing_candidate_fraction": float(np.mean(all_pass)),
            "constrained_paths_any_passing_candidate_fraction": (
                float(np.mean(constrained_pass)) if constrained_pass else None
            ),
            "n_paths": len(all_pass),
            "n_constrained_paths": len(constrained_pass),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction_csv", type=Path, required=True)
    parser.add_argument("--xyz_manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/dpa_ts"))
    parser.add_argument("--processed_folder", default="processed")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--accept_all",
        action="store_true",
        help="Copy every stereo-passing candidate instead of only the first per path.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    by_reaction_id, by_dpa_key = load_reactions(args.reaction_csv)
    constraint_cache = {}

    def constraints_for(row):
        reaction_id = row["reaction_id"].strip()
        if reaction_id not in constraint_cache:
            constraint_cache[reaction_id] = extract_reaction_constraints(
                row["rxn_smiles"]
            )
        return constraint_cache[reaction_id]

    manifest_rows = []
    for manifest in args.xyz_manifest:
        with manifest.open(newline="") as handle:
            manifest_rows.extend(csv.DictReader(handle))

    candidate_rows = []
    constraint_detail_rows = []
    mapping_failures = []
    for manifest_row in tqdm(manifest_rows, desc="Generated XYZ stereo checks"):
        reaction = reaction_for_path(
            manifest_row["path"], by_reaction_id, by_dpa_key
        )
        constraints, skipped, expected_numbers = constraints_for(reaction)
        xyz_path = Path(manifest_row["xyz_path"])
        numbers, coords = read_xyz(xyz_path)
        mapping_ok = bool(np.array_equal(numbers, expected_numbers))
        if not mapping_ok:
            mapping_failures.append(str(xyz_path))
        passed, details = evaluate_coords(coords, constraints)
        passed = bool(mapping_ok and passed)
        violations = [detail["constraint"] for detail in details if not detail["pass"]]
        candidate_row = dict(manifest_row)
        candidate_row.update(
            {
                "reaction_id": reaction["reaction_id"],
                "mapping_order_valid": int(mapping_ok),
                "n_constraints": len(constraints),
                "n_tetrahedral_constraints": sum(
                    constraint.kind == "tetrahedral" for constraint in constraints
                ),
                "n_double_bond_constraints": sum(
                    constraint.kind == "double_bond" for constraint in constraints
                ),
                "n_skipped_constraints": len(skipped),
                "n_violations": len(violations),
                "pass_all_stereo": int(passed),
                "violations": ";".join(violations),
            }
        )
        candidate_rows.append(candidate_row)
        for detail in details:
            constraint_detail_rows.append(
                {
                    "source": "generated",
                    "path": manifest_row["path"],
                    "rank": manifest_row["rank"],
                    "candidate_index": manifest_row["candidate_index"],
                    **detail,
                }
            )

    dataset = MoleculeDataset(
        root=str(args.dataset_root),
        processed_folder=args.processed_folder,
        split=args.split,
    )
    reference_rows = []
    for index, sample in enumerate(tqdm(dataset, desc="Reference TS stereo checks")):
        reaction_id = str(getattr(sample, "reaction_id", ""))
        if not reaction_id:
            key = tuple(
                normalized_id(getattr(sample, name))
                for name in ("ene_id", "diene_id", "prod_id")
            )
            reaction = by_dpa_key[key]
            path_key = "dpa:" + "_".join(key)
        else:
            reaction = by_reaction_id[reaction_id]
            path_key = f"reaction_id:{reaction_id}"
        constraints, skipped, expected_numbers = constraints_for(reaction)
        numbers = sample.numbers.detach().cpu().numpy().astype(np.int64)
        coords = sample.ts_coord.detach().cpu().numpy().astype(np.float64)
        mapping_ok = bool(np.array_equal(numbers, expected_numbers))
        passed, details = evaluate_coords(coords, constraints)
        passed = bool(mapping_ok and passed)
        violations = [detail["constraint"] for detail in details if not detail["pass"]]
        reference_rows.append(
            {
                "path": path_key,
                "dataset_index": index,
                "conf_id": str(getattr(sample, "conf_id", "")),
                "mapping_order_valid": int(mapping_ok),
                "n_constraints": len(constraints),
                "n_skipped_constraints": len(skipped),
                "n_violations": len(violations),
                "pass_all_stereo": int(passed),
                "violations": ";".join(violations),
            }
        )
        for detail in details:
            constraint_detail_rows.append(
                {
                    "source": "reference",
                    "path": path_key,
                    "rank": "",
                    "candidate_index": index,
                    **detail,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_csv = args.output_dir / "generated_xyz_stereo_checks.csv"
    reference_csv = args.output_dir / "reference_stereo_checks.csv"
    detail_csv = args.output_dir / "constraint_details.csv"
    for path, rows in (
        (candidate_csv, candidate_rows),
        (reference_csv, reference_rows),
        (detail_csv, constraint_detail_rows),
    ):
        if rows:
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    generated_by_path = defaultdict(list)
    for row in candidate_rows:
        generated_by_path[row["path"]].append(row)
    accepted_dir = args.output_dir / ("accepted_xyz" if args.accept_all else "accepted_top1_xyz")
    accepted_dir.mkdir(exist_ok=True)
    accepted_rows = []
    rejected_path_rows = []
    for path_key, rows in sorted(generated_by_path.items()):
        rows.sort(key=lambda row: int(row["rank"]))
        passing = [row for row in rows if int(row["pass_all_stereo"])]
        if passing:
            selected_rows = passing if args.accept_all else passing[:1]
            for selected_row in selected_rows:
                selected = dict(selected_row)
                source_path = Path(selected["xyz_path"])
                accepted_path = accepted_dir / source_path.name
                shutil.copy2(source_path, accepted_path)
                selected["selected_xyz_path"] = str(accepted_path.resolve())
                accepted_rows.append(selected)
        else:
            rejected_path_rows.append(
                {
                    "path": path_key,
                    "n_ranked_candidates_checked": len(rows),
                    "violated_constraints": ";".join(
                        sorted(
                            {
                                constraint
                                for row in rows
                                for constraint in str(row["violations"]).split(";")
                                if constraint
                            }
                        )
                    ),
                }
            )
    accepted_manifest = args.output_dir / "accepted_top1_manifest.csv"
    rejected_paths_csv = args.output_dir / "rejected_paths.csv"
    for path, rows in (
        (accepted_manifest, accepted_rows),
        (rejected_paths_csv, rejected_path_rows),
    ):
        if rows:
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    generated_constrained = [
        row for row in candidate_rows if int(row["n_constraints"]) > 0
    ]
    reference_constrained = [
        row for row in reference_rows if int(row["n_constraints"]) > 0
    ]
    reference_path_pass = defaultdict(bool)
    for row in reference_rows:
        reference_path_pass[row["path"]] |= bool(int(row["pass_all_stereo"]))
    reference_feasible_paths = {
        path for path, passed in reference_path_pass.items() if passed
    }
    generated_reference_feasible = [
        row for row in candidate_rows if row["path"] in reference_feasible_paths
    ]
    constraint_definition_count = sum(
        len(constraints) for constraints, _, _ in constraint_cache.values()
    )
    kind_counts = defaultdict(int)
    for constraints, _, _ in constraint_cache.values():
        for constraint in constraints:
            kind_counts[f"{constraint.side}_{constraint.kind}"] += 1
    skipped_constraints = [
        skipped
        for _, skipped_list, _ in constraint_cache.values()
        for skipped in skipped_list
    ]
    generated_details = [
        row for row in constraint_detail_rows if row["source"] == "generated"
    ]
    reference_details = [
        row for row in constraint_detail_rows if row["source"] == "reference"
    ]
    summary = {
        "reaction_csv": str(args.reaction_csv.resolve()),
        "xyz_manifests": [str(path.resolve()) for path in args.xyz_manifest],
        "definition": {
            "atom_index": "atom_map - 1; maps must be contiguous 1..N",
            "tetrahedral": (
                "normalized det([p1-p4,p2-p4,p3-p4]); neighbors ordered by "
                "descending RDKit CIP rank; R expects negative, S positive"
            ),
            "double_bond": (
                "normalized dot product of the two mapped RDKit stereo-atom vectors "
                "after projection perpendicular to the double-bond axis; E expects "
                "negative, Z positive relative to that ordered stereo-atom pair"
            ),
            "rejection": (
                "reject only when expected_sign * value < 0; zero, near-zero, "
                "undefined, and skipped constraints do not trigger rejection"
            ),
            "nitrogen_policy": (
                "ignore tetrahedral stereocenters centered on N and E/Z bonds with "
                "N at either stereogenic endpoint; N substituents on non-N centers "
                "remain constrained"
            ),
        },
        "mapping_order": {
            "generated_total": len(candidate_rows),
            "generated_valid": sum(
                int(row["mapping_order_valid"]) for row in candidate_rows
            ),
            "reference_total": len(reference_rows),
            "reference_valid": sum(
                int(row["mapping_order_valid"]) for row in reference_rows
            ),
            "failure_examples": mapping_failures[:10],
        },
        "constraints": {
            "n_reactions_checked": len(constraint_cache),
            "n_constraint_definitions": constraint_definition_count,
            "occurrences_by_kind_and_side": dict(kind_counts),
            "n_skipped": len(skipped_constraints),
            "skipped_examples": skipped_constraints[:20],
        },
        "generated": {
            "n_candidates": len(candidate_rows),
            "n_constrained_candidates": len(generated_constrained),
            "all_candidate_pass_fraction": float(
                np.mean([int(row["pass_all_stereo"]) for row in candidate_rows])
            ),
            "constrained_candidate_pass_fraction": (
                float(
                    np.mean(
                        [int(row["pass_all_stereo"]) for row in generated_constrained]
                    )
                )
                if generated_constrained
                else None
            ),
            "constraint_occurrence_pass_fraction": float(
                np.mean([int(row["pass"]) for row in generated_details])
            )
            if generated_details
            else None,
            "path_topk": path_topk_summary(candidate_rows, sorted(set(args.top_k))),
            "path_topk_on_reference_feasible_paths": path_topk_summary(
                generated_reference_feasible, sorted(set(args.top_k))
            ),
        },
        "reference": {
            "n_structures": len(reference_rows),
            "n_constrained_structures": len(reference_constrained),
            "all_structure_pass_fraction": float(
                np.mean([int(row["pass_all_stereo"]) for row in reference_rows])
            ),
            "constrained_structure_pass_fraction": (
                float(
                    np.mean(
                        [int(row["pass_all_stereo"]) for row in reference_constrained]
                    )
                )
                if reference_constrained
                else None
            ),
            "constraint_occurrence_pass_fraction": float(
                np.mean([int(row["pass"]) for row in reference_details])
            )
            if reference_details
            else None,
            "path_any_conformer_pass_fraction": float(
                np.mean(list(reference_path_pass.values()))
            ),
            "n_paths": len(reference_path_pass),
            "n_paths_with_any_passing_conformer": len(reference_feasible_paths),
            "n_paths_with_no_passing_conformer": (
                len(reference_path_pass) - len(reference_feasible_paths)
            ),
        },
        "value_distributions": {
            source: {
                kind: metric_summary(
                    [
                        row["signed_margin"]
                        for row in constraint_detail_rows
                        if row["source"] == source
                        and row["kind"] == kind
                        and np.isfinite(row["signed_margin"])
                    ]
                )
                for kind in ("tetrahedral", "double_bond")
            }
            for source in ("generated", "reference")
        },
        "constraint_status_counts": {
            source: {
                status: sum(
                    row["source"] == source and row["status"] == status
                    for row in constraint_detail_rows
                )
                for status in ("same_sign", "opposite", "zero", "undefined")
            }
            for source in ("generated", "reference")
        },
        "outputs": {
            "generated_checks_csv": str(candidate_csv.resolve()),
            "reference_checks_csv": str(reference_csv.resolve()),
            "constraint_details_csv": str(detail_csv.resolve()),
            "accepted_top1_xyz_dir": str(accepted_dir.resolve()),
            "accepted_top1_manifest_csv": str(accepted_manifest.resolve()),
            "accept_all": bool(args.accept_all),
            "accepted_top1_count": len(accepted_rows),
            "rejected_paths_csv": str(rejected_paths_csv.resolve()),
            "rejected_path_count": len(rejected_path_rows),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
