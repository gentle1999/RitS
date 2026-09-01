"""Prepare DPA transition-state XYZ files for RitS training.

The DPA corpus stores one XYZ file per optimized conformer.  The filename
contains ``ene_id``, ``diene_id`` and ``prod_id`` while the mapped reaction
SMILES live in the reaction CSV.  This script joins those two sources and
writes the PyG files consumed by :class:`megalodon.data.MoleculeDataset`.

Unlike the historical TS1x preparer, this script does not rely on row order
between the two inputs and keeps all conformers of one reaction in one split.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from tqdm import tqdm

try:
    from data_processing.prepare_ts1x_for_training import process_reaction
except ModuleNotFoundError:  # direct ``python data_processing/script.py`` execution
    from prepare_ts1x_for_training import process_reaction


XYZ_NAME_RE = re.compile(
    r"^(?P<ene>\d+)_(?P<diene>\d+)_(?P<prod>\d+)_conf_(?P<conf>\d+)"
    r"_ts\.(?P<digest>[0-9a-fA-F]+)\.xyz$"
)
ATOMIC_NUMBERS = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}


def _key(value: str) -> str:
    """Normalize CSV/filename numeric IDs without losing their identity."""
    return str(int(str(value).strip()))


def _reaction_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (_key(row["ene_id"]), _key(row["diene_id"]), _key(row["prod_id"]))


def _parse_xyz(path: Path) -> tuple[str, list[str], np.ndarray, int]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ file has fewer than two header lines")
    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"invalid atom count {lines[0]!r}") from exc
    atom_lines = lines[2:]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"expected {n_atoms} atom lines, found {len(atom_lines)}")

    symbols: list[str] = []
    coords: list[list[float]] = []
    for line_no, line in enumerate(atom_lines, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"line {line_no} has fewer than 4 fields")
        symbol = fields[0]
        if symbol not in ATOMIC_NUMBERS:
            raise ValueError(f"unsupported element {symbol!r} on line {line_no}")
        symbols.append(symbol)
        try:
            coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
        except ValueError as exc:
            raise ValueError(f"invalid coordinates on line {line_no}") from exc

    charge = 0
    # DPA comments are normally ``comment charge 0 multiplicity 1``.  Keep
    # parsing deliberately permissive so a missing/non-standard comment still
    # defaults to the neutral charge used by the source reaction table.
    comment = lines[1].lower()
    charge_match = re.search(r"\bcharge\s+(-?\d+)", comment)
    if charge_match:
        charge = int(charge_match.group(1))
    return "\n".join(lines), symbols, np.asarray(coords, dtype=np.float32), charge


def _mapped_atomic_numbers(reaction_smiles: str) -> list[int]:
    reactants, products = reaction_smiles.split(">>")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    r = Chem.MolFromSmiles(reactants, params)
    p = Chem.MolFromSmiles(products, params)
    if r is None or p is None:
        raise ValueError("RDKit could not parse mapped reaction SMILES")
    if r.GetNumAtoms() != p.GetNumAtoms():
        raise ValueError("reactant/product atom counts differ")
    r_maps = [a.GetAtomMapNum() for a in r.GetAtoms()]
    p_maps = [a.GetAtomMapNum() for a in p.GetAtoms()]
    n_atoms = r.GetNumAtoms()
    expected = list(range(1, n_atoms + 1))
    if sorted(r_maps) != expected or sorted(p_maps) != expected:
        raise ValueError("atom maps must be contiguous and start at 1")
    r_by_map = {a.GetAtomMapNum(): a.GetAtomicNum() for a in r.GetAtoms()}
    p_by_map = {a.GetAtomMapNum(): a.GetAtomicNum() for a in p.GetAtoms()}
    if r_by_map != p_by_map:
        raise ValueError("reactant/product atomic numbers differ by atom map")
    return [r_by_map[i] for i in expected]


def _file_sort_key(path: Path) -> tuple[int, int, str]:
    match = XYZ_NAME_RE.match(path.name)
    assert match is not None
    return (int(match.group("conf")), int(match.group("prod")), path.name)


def _save_dataset(items: list[Data], path: Path) -> None:
    if not items:
        raise ValueError(f"cannot save an empty split: {path}")
    batch = collate(items[0].__class__, items, increment=False, add_batch=False)
    torch.save(batch[:2], path)


def _load_excluded_files(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return set()
    columns = set(rows[0])
    if "file" in columns:
        return {Path(row["file"]).name for row in rows if row["file"].strip()}
    if "source_xyz" in columns:
        return {
            Path(row["source_xyz"]).name
            for row in rows
            if row["source_xyz"].strip()
        }
    raise ValueError("exclusion manifest must contain a 'file' or 'source_xyz' column")


def _split_groups(
    keys: list[tuple[str, str, str]], train_ratio: float, val_ratio: float, seed: int
) -> dict[tuple[str, str, str], str]:
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to less than 1")
    unique = sorted(set(keys))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_train = int(len(unique) * train_ratio)
    n_val = int(len(unique) * val_ratio)
    return {
        key: "train" if i < n_train else "val" if i < n_train + n_val else "test"
        for i, key in enumerate(unique)
    }


def prepare(
    reaction_csv: Path,
    xyz_dir: Path,
    save_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    add_stereo: bool = True,
    exclude_manifest: Path | None = None,
) -> dict:
    with reaction_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"ene_id", "diene_id", "prod_id", "rxn_smiles"}
    missing_columns = required - set(rows[0]) if rows else required
    if missing_columns:
        raise ValueError(f"reaction CSV missing columns: {sorted(missing_columns)}")

    reactions: dict[tuple[str, str, str], dict[str, str]] = {}
    duplicate_keys = []
    for row in rows:
        key = _reaction_key(row)
        if key in reactions:
            duplicate_keys.append(key)
        reactions[key] = row
    if duplicate_keys:
        raise ValueError(f"reaction CSV contains duplicate reaction keys, e.g. {duplicate_keys[:3]}")

    excluded_files = _load_excluded_files(exclude_manifest)
    xyz_by_key: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    malformed_files: list[str] = []
    for path in sorted(xyz_dir.glob("*.xyz")):
        match = XYZ_NAME_RE.match(path.name)
        if match is None:
            malformed_files.append(path.name)
            continue
        key = tuple(_key(match.group(part)) for part in ("ene", "diene", "prod"))
        xyz_by_key[key].append(path)
    for paths in xyz_by_key.values():
        paths.sort(key=_file_sort_key)

    matched_keys = sorted(set(reactions) & set(xyz_by_key))
    missing_xyz_keys = sorted(set(reactions) - set(xyz_by_key))
    orphan_xyz_keys = sorted(set(xyz_by_key) - set(reactions))
    if not matched_keys:
        raise ValueError("no XYZ files matched reaction CSV keys")

    split_by_key = _split_groups(matched_keys, train_ratio, val_ratio, seed)
    input_xyz_files = {path.name for paths in xyz_by_key.values() for path in paths}
    missing_excluded_files = sorted(excluded_files - input_xyz_files)
    retained_xyz_by_key = {
        key: [path for path in paths if path.name not in excluded_files]
        for key, paths in xyz_by_key.items()
    }
    exhausted_keys = sorted(
        key for key in matched_keys if not retained_xyz_by_key.get(key)
    )
    datasets: dict[str, list[Data]] = {"train": [], "val": [], "test": []}
    metadata: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    atom_order_checks = 0

    for key in tqdm(matched_keys, desc="Preparing DPA TS"):
        row = reactions[key]
        reaction_smiles = row["rxn_smiles"]
        try:
            expected_numbers = _mapped_atomic_numbers(reaction_smiles)
        except Exception as exc:
            failures.append({"key": "_".join(key), "file": "", "error": str(exc)})
            continue
        split = split_by_key[key]
        for xyz_path in retained_xyz_by_key[key]:
            try:
                xyz_block, symbols, _coords, charge = _parse_xyz(xyz_path)
                observed_numbers = [ATOMIC_NUMBERS[s] for s in symbols]
                if observed_numbers != expected_numbers:
                    raise ValueError(
                        "XYZ atom order/elements do not match mapped reaction order "
                        f"(expected {len(expected_numbers)} atoms, got {len(observed_numbers)})"
                    )
                data = process_reaction(
                    *reaction_smiles.split(">>"),
                    xyz_block,
                    kekulize=True,
                    add_stereo=add_stereo,
                )
                data.charges = torch.full_like(data.numbers, charge, dtype=torch.int8)
                match = XYZ_NAME_RE.match(xyz_path.name)
                assert match is not None
                data.id = xyz_path.stem
                data.ene_id = key[0]
                data.diene_id = key[1]
                data.prod_id = key[2]
                data.conf_id = match.group("conf")
                data.reaction_id = row.get("reaction_id", "")
                datasets[split].append(data)
                metadata.append(
                    {
                        "file": xyz_path.name,
                        "ene_id": key[0],
                        "diene_id": key[1],
                        "prod_id": key[2],
                        "conf_id": match.group("conf"),
                        "reaction_id": row.get("reaction_id", ""),
                        "split": split,
                        "rxn_smiles": reaction_smiles,
                    }
                )
                atom_order_checks += 1
            except Exception as exc:
                failures.append({"key": "_".join(key), "file": xyz_path.name, "error": str(exc)})

    processed = save_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for split, items in datasets.items():
        _save_dataset(items, processed / f"{split}_h.pt")

    metadata_path = save_dir / "metadata.csv"
    metadata_fields = [
        "file", "ene_id", "diene_id", "prod_id", "conf_id", "reaction_id", "split", "rxn_smiles"
    ]
    with metadata_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_fields)
        writer.writeheader()
        writer.writerows(metadata)

    archived_exclude_manifest = None
    if exclude_manifest is not None:
        archived_exclude_manifest = save_dir / "excluded_references.csv"
        if exclude_manifest.resolve() != archived_exclude_manifest.resolve():
            shutil.copy2(exclude_manifest, archived_exclude_manifest)

    summary = {
        "reaction_csv": str(reaction_csv),
        "xyz_dir": str(xyz_dir),
        "save_dir": str(save_dir),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "reaction_rows": len(rows),
        "reaction_keys": len(reactions),
        "xyz_files": sum(len(v) for v in xyz_by_key.values()),
        "exclude_manifest": str(exclude_manifest) if exclude_manifest else None,
        "archived_exclude_manifest": (
            str(archived_exclude_manifest) if archived_exclude_manifest else None
        ),
        "excluded_files_requested": len(excluded_files),
        "excluded_files_found": len(excluded_files) - len(missing_excluded_files),
        "excluded_files_missing": missing_excluded_files,
        "reaction_keys_with_no_retained_files": ["_".join(k) for k in exhausted_keys],
        "retained_xyz_files": sum(len(v) for v in retained_xyz_by_key.values()),
        "matched_reaction_keys": len(matched_keys),
        "missing_xyz_keys": len(missing_xyz_keys),
        "orphan_xyz_keys": len(orphan_xyz_keys),
        "malformed_xyz_files": len(malformed_files),
        "atom_order_checks": atom_order_checks,
        "processed_files": len(metadata),
        "failed_files": len(failures),
        "split_reaction_keys": Counter(split_by_key.values()),
        "split_files": {name: len(items) for name, items in datasets.items()},
        "missing_xyz_key_values": ["_".join(k) for k in missing_xyz_keys],
        "orphan_xyz_key_values": ["_".join(k) for k in orphan_xyz_keys],
        "malformed_xyz_file_values": malformed_files,
        "failures": failures,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True, default=dict) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-csv", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-stereo", action="store_true", help="Do not encode stereo bonds")
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        help="CSV with a file or source_xyz column listing conformers to omit.",
    )
    args = parser.parse_args()
    summary = prepare(
        reaction_csv=args.reaction_csv,
        xyz_dir=args.xyz_dir,
        save_dir=args.save_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        add_stereo=not args.no_stereo,
        exclude_manifest=args.exclude_manifest,
    )
    print(json.dumps({k: summary[k] for k in (
        "reaction_rows", "xyz_files", "matched_reaction_keys", "processed_files",
        "failed_files", "missing_xyz_keys", "split_files"
    )}, indent=2, default=dict))


if __name__ == "__main__":
    main()
