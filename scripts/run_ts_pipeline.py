#!/usr/bin/env python3
"""Generate, cluster, and rank TS candidates with the g-xTB backend.

The generation stage delegates to ``generate_reaction_queue.py``.  The
``hybrid`` strategy uses reaction-major queue ordering plus an atom budget:
all seeds of one reaction stay adjacent, while several reaction groups can
share a GPU batch.  This keeps the graph locality of the old rxn-level
generator and the resumability/load balancing of the global queue.

The g-xTB project currently distributes a modified ``xtb`` executable rather
than a ``tblite.interface.Calculator`` method.  Therefore energy evaluation
uses the official executable with ``--gxtb``.  The command is explicit in the
output metadata so GFN1/GFN2 calculations cannot be confused with g-xTB.

The default ``staged`` energy workflow evaluates cluster medoids first, filters
clusters by a per-reaction energy window, then evaluates every member of the
surviving clusters and selects the lowest-energy member of each cluster.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

from check_ts_stereo_constraints import (
    evaluate_coords as evaluate_stereo_coords,
    extract_reaction_constraints,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATOR = PROJECT_ROOT / "scripts" / "generate_reaction_queue.py"
ANGSTROM_TO_BOHR = 1.8897259886
HARTREE_TO_KCAL_MOL = 627.509474
ENERGY_RE = re.compile(r"TOTAL ENERGY\s+([-+0-9.Ee]+)\s+Eh", re.IGNORECASE)
GRADIENT_RE = re.compile(
    r"GRADIENT NORM\s+([-+0-9.Ee]+)\s+Eh", re.IGNORECASE
)
SEED_RE = re.compile(r"^seed_(\d+)$")
RXN_RE = re.compile(r"^rxn_(\d+)\.xyz$")


def read_reactions(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "rxn_smiles" not in reader.fieldnames:
                raise ValueError(f"CSV reaction input must contain an rxn_smiles column: {path}")
            reactions = [
                (row.get("rxn_smiles") or "").strip()
                for row in reader
            ]
        return [reaction for reaction in reactions if reaction]
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def read_xyz(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    lines = path.read_text().splitlines()
    if len(lines) < 2:
        raise ValueError(f"XYZ is truncated: {path}")
    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"invalid atom count in {path}") from exc
    if len(lines) != n_atoms + 2:
        raise ValueError(
            f"XYZ line count mismatch in {path}: expected {n_atoms + 2}, "
            f"got {len(lines)}"
        )
    comment = lines[1]
    numbers = []
    coords = []
    periodic = {
        symbol: number
        for number, symbol in enumerate(
            "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr "
            "Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh "
            "Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy "
            "Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
            "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split(),
            start=1,
        )
    }
    for line in lines[2:]:
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"invalid coordinate row in {path}: {line!r}")
        symbol = fields[0]
        if symbol not in periodic:
            raise ValueError(f"unknown element {symbol!r} in {path}")
        numbers.append(periodic[symbol])
        coords.append([float(value) for value in fields[1:]])
    return (
        np.asarray(numbers, dtype=np.int64),
        np.asarray(coords, dtype=np.float64),
        comment,
    )


def reaction_key(rxn_smiles: str) -> str:
    return hashlib.sha1(rxn_smiles.encode("utf-8")).hexdigest()[:16]


def infer_reaction_charge(rxn_smiles: str) -> int:
    from rdkit import Chem

    sides = rxn_smiles.split(">>")
    if len(sides) != 2:
        raise ValueError(f"invalid reaction SMILES: {rxn_smiles}")
    charges = []
    for side in sides:
        molecule = Chem.MolFromSmiles(side, sanitize=False)
        if molecule is None:
            raise ValueError(f"cannot parse reaction side for charge: {side}")
        charges.append(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()))
    if charges[0] != charges[1]:
        raise ValueError(
            f"reactant/product total charge differs in reaction: {charges}"
        )
    return int(charges[0])


def collect_candidates(
    candidates_dir: Path,
    reactions_path: Path | None,
    expected_samples: int | None,
    max_reactions: int | None = None,
    start_reaction: int = 0,
    end_reaction: int | None = None,
) -> list[dict]:
    """Collect seed XYZ files and group them by their exact comment reaction."""
    manifest = read_reactions(reactions_path) if reactions_path else None
    manifest_index = {reaction: index for index, reaction in enumerate(manifest or [])}
    if start_reaction < 0:
        raise ValueError("start_reaction must be non-negative")
    if end_reaction is not None and end_reaction < start_reaction:
        raise ValueError("end_reaction must be greater than or equal to start_reaction")
    if max_reactions is not None and max_reactions < 0:
        raise ValueError("max_reactions must be non-negative")
    range_stop = None if end_reaction is None else max(start_reaction, end_reaction)
    if manifest is not None:
        range_stop = len(manifest) if range_stop is None else min(range_stop, len(manifest))
    if max_reactions is not None:
        range_stop = start_reaction + max_reactions if range_stop is None else min(
            range_stop, start_reaction + max_reactions
        )
    allowed_indices = None if range_stop is None else set(range(start_reaction, range_stop))
    grouped: dict[str, dict] = {}
    for path in sorted(candidates_dir.glob("seed_*/*.xyz")):
        seed_match = SEED_RE.match(path.parent.name)
        rxn_match = RXN_RE.match(path.name)
        if seed_match is None or rxn_match is None:
            continue
        seed = int(seed_match.group(1))
        rxn_index = int(rxn_match.group(1))
        if allowed_indices is not None and rxn_index not in allowed_indices:
            continue
        numbers, coords, comment = read_xyz(path)
        if not comment.startswith("rxn_smiles="):
            raise ValueError(f"missing rxn_smiles comment: {path}")
        rxn_smiles = comment[len("rxn_smiles=") :]
        if manifest is not None and rxn_smiles not in manifest_index:
            raise ValueError(f"XYZ comment is absent from reaction manifest: {path}")
        if manifest is not None and manifest_index[rxn_smiles] != rxn_index:
            raise ValueError(
                f"reaction index/comment mismatch in {path}: "
                f"manifest index is {manifest_index[rxn_smiles]}"
            )
        key = rxn_smiles
        record = grouped.setdefault(
            key,
            {
                "reaction_index": manifest_index.get(rxn_smiles, rxn_index),
                "rxn_smiles": rxn_smiles,
                "reaction_key": reaction_key(rxn_smiles),
                "charge": infer_reaction_charge(rxn_smiles),
                "numbers": numbers,
                "candidates": [],
            },
        )
        if not np.array_equal(record["numbers"], numbers):
            raise ValueError(f"atom order/elements differ within reaction: {rxn_smiles}")
        record["candidates"].append(
            {
                "seed": seed,
                "path": path,
                "coords": coords,
                "rxn_index_from_name": rxn_index,
            }
        )

    records = []
    for record in grouped.values():
        record["candidates"].sort(key=lambda item: (item["seed"], str(item["path"])))
        seeds = [candidate["seed"] for candidate in record["candidates"]]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed within reaction {record['reaction_index']}")
        if expected_samples is not None and len(record["candidates"]) != expected_samples:
            raise ValueError(
                f"reaction {record['reaction_index']} has {len(record['candidates'])} "
                f"candidates, expected {expected_samples}"
            )
        records.append(record)
    records.sort(key=lambda item: (item["reaction_index"], item["rxn_smiles"]))
    if manifest is not None:
        expected_manifest = manifest[
            start_reaction : range_stop if range_stop is not None else len(manifest)
        ]
        missing = [reaction for reaction in expected_manifest if reaction not in grouped]
        if missing:
            raise ValueError(f"candidate directory is missing {len(missing)} reactions")
    if not records:
        raise ValueError(f"no seed XYZ files found below {candidates_dir}")
    if max_reactions is not None:
        records = records[:max_reactions]
    return records


def filter_stereo_candidates(records: list[dict]) -> tuple[list[dict], dict]:
    """Apply mapped tetrahedral/E-Z sign checks before any clustering."""
    audit_rows = []
    stats = {
        "enabled": True,
        "n_raw_candidates": 0,
        "n_stereo_valid_candidates": 0,
        "n_stereo_rejected_candidates": 0,
        "n_reactions_without_stereo_candidates": 0,
        "n_constraints": 0,
        "n_skipped_constraints": 0,
    }
    for record in records:
        constraints, skipped, expected_numbers = extract_reaction_constraints(
            record["rxn_smiles"]
        )
        mapping_ok = bool(np.array_equal(record["numbers"], expected_numbers))
        stats["n_constraints"] += len(constraints)
        stats["n_skipped_constraints"] += len(skipped)
        original_candidates = record["candidates"]
        kept = []
        for source_candidate_index, candidate in enumerate(original_candidates):
            passed, details = evaluate_stereo_coords(
                candidate["coords"], constraints
            )
            passed = bool(mapping_ok and passed)
            violations = [
                detail["constraint"] for detail in details if not detail["pass"]
            ]
            audit_rows.append(
                {
                    "reaction_index": record["reaction_index"],
                    "seed": candidate["seed"],
                    "candidate_index": source_candidate_index,
                    "source_xyz": str(candidate["path"].resolve()),
                    "mapping_order_valid": int(mapping_ok),
                    "n_constraints": len(constraints),
                    "n_skipped_constraints": len(skipped),
                    "pass_all_stereo": int(passed),
                    "violations": ";".join(violations),
                }
            )
            candidate["source_candidate_index"] = source_candidate_index
            if passed:
                kept.append(candidate)
        record["n_raw_candidates"] = len(original_candidates)
        record["n_stereo_valid_candidates"] = len(kept)
        record["n_stereo_rejected_candidates"] = len(original_candidates) - len(kept)
        record["candidates"] = kept
        stats["n_raw_candidates"] += len(original_candidates)
        stats["n_stereo_valid_candidates"] += len(kept)
        stats["n_stereo_rejected_candidates"] += len(original_candidates) - len(kept)
        if not kept:
            stats["n_reactions_without_stereo_candidates"] += 1
    return audit_rows, stats


def pairwise_kabsch_rmsd(coords: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a complete pairwise Kabsch RMSD matrix."""
    xyz = coords[:, mask, :].astype(np.float64, copy=True)
    xyz -= xyz.mean(axis=1, keepdims=True)
    left, right = np.triu_indices(len(xyz), k=1)
    matrix = np.zeros((len(xyz), len(xyz)), dtype=np.float64)
    if len(left) == 0:
        return matrix
    p = xyz[left]
    q = xyz[right]
    covariance = np.einsum("pai,paj->pij", p, q)
    u, _, vh = np.linalg.svd(covariance)
    correction = np.ones((len(left), 3), dtype=np.float64)
    correction[:, -1] = np.sign(np.linalg.det(np.matmul(u, vh)))
    rotation = np.matmul(u * correction[:, None, :], vh)
    aligned = np.matmul(p, rotation)
    values = np.sqrt(np.mean(np.sum((aligned - q) ** 2, axis=-1), axis=-1))
    matrix[left, right] = values
    matrix[right, left] = values
    return matrix


def cluster_labels(distance_matrix: np.ndarray, threshold: float) -> np.ndarray:
    if len(distance_matrix) == 1:
        return np.ones(1, dtype=np.int64)
    condensed = squareform(distance_matrix, checks=False)
    return fcluster(linkage(condensed, method="complete"), t=threshold, criterion="distance")


def cluster_medoids(labels: np.ndarray, distance_matrix: np.ndarray) -> dict[int, int]:
    medoids = {}
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        means = distance_matrix[np.ix_(members, members)].mean(axis=1)
        medoids[int(label)] = int(members[np.argmin(means)])
    return medoids


def write_xyz(path: Path, numbers: np.ndarray, coords: np.ndarray, comment: str) -> None:
    symbols = {
        1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
        17: "Cl", 35: "Br", 53: "I",
    }
    # RDKit is used for uncommon elements in the generator; this fallback is
    # only for selected outputs and keeps the standard first-row elements clear.
    try:
        from rdkit import Chem
        get_symbol = lambda number: Chem.GetPeriodicTable().GetElementSymbol(int(number))
    except ImportError:
        get_symbol = lambda number: symbols[int(number)]
    lines = [str(len(numbers)), comment]
    for number, (x, y, z) in zip(numbers, coords):
        lines.append(f"{get_symbol(number):<3s} {x: .10f} {y: .10f} {z: .10f}")
    path.write_text("\n".join(lines) + "\n")


def parse_energy(output: str) -> tuple[float | None, float | None]:
    energies = ENERGY_RE.findall(output)
    gradients = GRADIENT_RE.findall(output)
    energy = float(energies[-1]) if energies else None
    gradient = float(gradients[-1]) if gradients else None
    return energy, gradient


def resolve_gxtb_binary(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    discovered = shutil.which("gxtb")
    if discovered:
        candidates.append(Path(discovered))
    environment_binary = Path(sys.executable).resolve().parent / "gxtb"
    candidates.append(environment_binary)
    binary = next((Path(candidate).resolve() for candidate in candidates if Path(candidate).is_file()), None)
    if binary is None:
        raise FileNotFoundError(
            "g-xTB binary not found; install the official g-xtb binary and "
            "pass --gxtb-binary /path/to/xtb"
        )
    probe = subprocess.run(
        [str(binary), "--help"], capture_output=True, text=True, check=False,
        timeout=30,
    )
    if "--gxtb" not in probe.stdout + probe.stderr:
        raise ValueError(
            f"{binary} is not the official g-xTB-enabled xtb binary: "
            "its help output does not contain --gxtb"
        )
    return binary


def run_gxtb(task, binary: Path, work_root: Path, timeout: float, omp_threads: int) -> dict:
    reaction_index, candidate_index, numbers, coords, charge, uhf = task
    started = time.perf_counter()
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(omp_threads),
            "MKL_NUM_THREADS": str(omp_threads),
            "OPENBLAS_NUM_THREADS": str(omp_threads),
        }
    )
    with tempfile.TemporaryDirectory(dir=work_root) as temporary:
        xyz_path = Path(temporary) / "candidate.xyz"
        write_xyz(xyz_path, numbers, coords, f"reaction_index={reaction_index} candidate={candidate_index}")
        command = [
            str(binary), xyz_path.name, "--gxtb", "--sp",
            "--chrg", str(charge),
        ]
        if uhf is not None:
            command += ["--uhf", str(uhf)]
        try:
            result = subprocess.run(
                command, cwd=temporary, env=env, capture_output=True,
                text=True, timeout=timeout, check=False,
            )
            output = result.stdout + "\n" + result.stderr
            energy, gradient = parse_energy(output)
            success = result.returncode == 0 and energy is not None
            error = "" if success else output[-1200:].replace("\n", " ")
        except subprocess.TimeoutExpired:
            success, energy, gradient = False, None, None
            error = f"timeout after {timeout}s"
        except OSError as exc:
            success, energy, gradient = False, None, None
            error = repr(exc)
    return {
        "reaction_index": reaction_index,
        "candidate_index": candidate_index,
        "charge": charge,
        "uhf": "" if uhf is None else uhf,
        "success": int(success),
        "energy_hartree": energy,
        "gradient_norm_hartree_per_bohr": gradient,
        "wall_seconds": time.perf_counter() - started,
        "error": error,
    }


ENERGY_FIELDS = [
    "reaction_index", "candidate_index", "charge", "uhf", "success", "energy_hartree",
    "gradient_norm_hartree_per_bohr", "wall_seconds", "error",
]


def load_energy_cache(path: Path) -> dict[tuple[int, int], dict]:
    if not path.exists():
        return {}
    cache = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                key = (int(row["reaction_index"]), int(row["candidate_index"]))
            except (KeyError, TypeError, ValueError):
                # Ignore rows from an older/incomplete cache schema; they will
                # be recalculated and written in the current schema.
                continue
            cache[key] = row
    return cache


def reusable_energy_row(row: dict, charge: int, uhf: int | None) -> bool:
    """Only reuse successful rows calculated with the requested spin settings."""
    try:
        success = int(row.get("success", 0))
    except (TypeError, ValueError):
        return False
    if success != 1:
        return False
    try:
        if int(row["charge"]) != int(charge):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    cached_uhf = str(row.get("uhf", ""))
    requested_uhf = "" if uhf is None else str(uhf)
    return cached_uhf == requested_uhf


def successful_energy(row: dict | None) -> float | None:
    """Return a cached energy, or None for a missing/failed/malformed row."""
    if not row:
        return None
    try:
        if int(row["success"]) != 1:
            return None
        return float(row["energy_hartree"])
    except (KeyError, TypeError, ValueError):
        return None


def append_energy_rows(path: Path, rows: list[dict]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENERGY_FIELDS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "energy_hartree": "" if row["energy_hartree"] is None else f"{row['energy_hartree']:.12f}",
                    "gradient_norm_hartree_per_bohr": "" if row["gradient_norm_hartree_per_bohr"] is None else f"{row['gradient_norm_hartree_per_bohr']:.12f}",
                    "wall_seconds": f"{row['wall_seconds']:.6f}",
                }
            )


def run_generation(args: argparse.Namespace) -> None:
    command = [
        sys.executable, str(GENERATOR),
        "--reactions", str(args.reactions),
        "--config", str(args.config),
        "--ckpt", str(args.ckpt),
        "--output_dir", str(args.candidates),
        "--n_samples", str(args.n_samples),
        "--start_reaction", str(args.start_reaction),
        "--num_steps", str(args.num_steps),
        "--device", args.device,
        "--kekulize", "--add_stereo", "--skip_existing",
        "--queue_order", "reaction-major" if args.generation_strategy == "hybrid" else "seed-major",
    ]
    if args.end_reaction is not None:
        command += ["--end_reaction", str(args.end_reaction)]
    if args.max_atoms_per_batch is not None:
        command += ["--max_atoms_per_batch", str(args.max_atoms_per_batch)]
    else:
        command += ["--batch_size", str(args.batch_size)]
    print("generation_command:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def evaluate(args: argparse.Namespace) -> dict:
    all_records = collect_candidates(
        args.candidates,
        args.reactions,
        args.n_samples,
        args.max_reactions,
        args.start_reaction,
        args.end_reaction,
    )
    for record in all_records:
        record.setdefault("n_raw_candidates", len(record["candidates"]))
        record.setdefault("n_stereo_valid_candidates", len(record["candidates"]))
        record.setdefault("n_stereo_rejected_candidates", 0)
    stereo_rows = []
    stereo_stats = {
        "enabled": bool(args.stereo_filter),
        "n_raw_candidates": sum(len(record["candidates"]) for record in all_records),
        "n_stereo_valid_candidates": sum(len(record["candidates"]) for record in all_records),
        "n_stereo_rejected_candidates": 0,
        "n_reactions_without_stereo_candidates": 0,
        "n_constraints": 0,
        "n_skipped_constraints": 0,
    }
    if args.stereo_filter:
        stereo_rows, stereo_stats = filter_stereo_candidates(all_records)
    records = [record for record in all_records if record["candidates"]]
    args.output.mkdir(parents=True, exist_ok=True)
    binary = resolve_gxtb_binary(args.gxtb_binary)
    energy_path = args.output / "energy_results.csv"
    energy_cache = load_energy_cache(energy_path)

    for record in tqdm(records, desc="clustering reactions"):
        coords = np.stack([candidate["coords"] for candidate in record["candidates"]])
        mask = record["numbers"] > 1 if args.cluster_atoms == "heavy" else record["numbers"] > 0
        distances = pairwise_kabsch_rmsd(coords, mask)
        labels = cluster_labels(distances, args.cluster_threshold)
        medoids = cluster_medoids(labels, distances)
        record["coords"] = coords
        record["labels"] = labels
        record["distances"] = distances
        record["medoids"] = medoids

    def energy_cache_key(record: dict, index: int) -> tuple[int, int]:
        candidate = record["candidates"][index]
        return (
            record["reaction_index"],
            int(candidate.get("source_candidate_index", index)),
        )

    def run_energy_stage(indices_by_record: dict[int, list[int]], description: str) -> None:
        """Evaluate the requested candidate indices, reusing successful cache rows."""
        tasks = []
        for record in records:
            indices = indices_by_record.get(record["reaction_index"], [])
            requested_charge = record["charge"] if args.charge is None else args.charge
            for index in indices:
                key = energy_cache_key(record, index)
                if not reusable_energy_row(
                    energy_cache.get(key, {}), requested_charge, args.uhf
                ):
                    tasks.append(
                        (
                            record["reaction_index"], key[1], record["numbers"].tolist(),
                            record["coords"][index].tolist(),
                            requested_charge,
                            args.uhf,
                        )
                    )
        if not tasks:
            return
        work_root = args.output / "gxtb_work"
        work_root.mkdir(exist_ok=True)
        pending = []
        with ThreadPoolExecutor(max_workers=args.energy_workers) as executor:
            futures = [
                executor.submit(
                    run_gxtb, task, binary, work_root,
                    args.energy_timeout, args.omp_threads,
                )
                for task in tasks
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=description):
                row = future.result()
                energy_cache[(row["reaction_index"], row["candidate_index"])] = row
                pending.append(row)
                if len(pending) >= 100:
                    append_energy_rows(energy_path, pending)
                    pending.clear()
        if pending:
            append_energy_rows(energy_path, pending)
        shutil.rmtree(work_root, ignore_errors=True)

    medoid_indices = {
        record["reaction_index"]: list(record["medoids"].values())
        for record in records
    }
    # Every mode starts with the cheap geometric-center pass.  The staged mode
    # then expands only clusters whose medoid is within the reaction-level
    # energy window of the best medoid.
    run_energy_stage(medoid_indices, "g-xTB medoid single points")

    if args.energy_scope == "staged":
        for record in records:
            medoid_energy_by_label = {
                int(label): successful_energy(
                    energy_cache.get(energy_cache_key(record, medoid))
                )
                for label, medoid in record["medoids"].items()
            }
            successful_medoids = [
                energy for energy in medoid_energy_by_label.values()
                if energy is not None
            ]
            stage1_min = min(successful_medoids, default=None)
            record["stage1_min_energy"] = stage1_min
            record["stage1_medoid_energy"] = medoid_energy_by_label
            record["stage1_passed_labels"] = {
                label for label, energy in medoid_energy_by_label.items()
                if energy is not None
                and stage1_min is not None
                and energy <= stage1_min + args.energy_window_hartree + 1e-12
            }
        expanded_indices = {}
        for record in records:
            labels = record["labels"]
            expanded_indices[record["reaction_index"]] = [
                index
                for label in record["stage1_passed_labels"]
                for index in np.flatnonzero(labels == label).tolist()
            ]
        run_energy_stage(expanded_indices, "g-xTB retained-cluster single points")
    else:
        for record in records:
            medoid_energy_by_label = {
                int(label): successful_energy(
                    energy_cache.get(energy_cache_key(record, medoid))
                )
                for label, medoid in record["medoids"].items()
            }
            successful_medoids = [
                energy for energy in medoid_energy_by_label.values()
                if energy is not None
            ]
            record["stage1_min_energy"] = min(successful_medoids, default=None)
            record["stage1_medoid_energy"] = medoid_energy_by_label
            record["stage1_passed_labels"] = {
                label for label, energy in medoid_energy_by_label.items()
                if energy is not None
            }

    if args.energy_scope == "all":
        # Exhaustive mode evaluates every candidate, but still reports the
        # medoid pass so the same cluster-level diagnostics are available.
        run_energy_stage(
            {
                record["reaction_index"]: list(range(len(record["candidates"])))
                for record in records
            },
            "g-xTB all-candidate single points",
        )

    candidate_rows = []
    cluster_rows = []
    reaction_rows = []
    selected_dir = args.output / "selected_xyz"
    selected_dir.mkdir(exist_ok=True)
    # Selected filenames are deterministic; remove only files owned by this
    # output so reruns with a different energy window cannot leave stale XYZs.
    for stale in selected_dir.glob("*.xyz"):
        stale.unlink()
    current_energy_keys = {
        energy_cache_key(record, index)
        for record in records
        for index in range(len(record["candidates"]))
    }
    for record in records:
        reaction_index = record["reaction_index"]
        labels = record["labels"]
        energy_by_index = {}
        for index in range(len(record["candidates"])):
            energy = successful_energy(energy_cache.get(energy_cache_key(record, index)))
            if energy is not None:
                energy_by_index[index] = energy
        representatives = []
        for label in sorted(np.unique(labels)):
            members = np.flatnonzero(labels == label).tolist()
            eligible = [index for index in members if index in energy_by_index]
            medoid = record["medoids"][int(label)]
            medoid_energy = record["stage1_medoid_energy"].get(int(label))
            stage1_min = record["stage1_min_energy"]
            medoid_delta = (
                "" if medoid_energy is None or stage1_min is None
                else f"{medoid_energy - stage1_min:.12f}"
            )
            passed_filter = int(int(label) in record["stage1_passed_labels"])
            if args.energy_scope == "staged" and not passed_filter:
                representative = None
            else:
                representative = min(eligible, key=energy_by_index.get) if eligible else None
            cluster_rows.append(
                {
                    "reaction_index": reaction_index,
                    "rxn_smiles": record["rxn_smiles"],
                    "cluster": int(label),
                    "cluster_size": len(members),
                    "medoid_candidate_index": medoid,
                    "medoid_seed": record["candidates"][medoid]["seed"],
                    "medoid_energy_hartree": "" if medoid_energy is None else f"{medoid_energy:.12f}",
                    "medoid_delta_energy_hartree": medoid_delta,
                    "passed_medoid_energy_filter": passed_filter,
                    "representative_candidate_index": "" if representative is None else representative,
                    "representative_seed": "" if representative is None else record["candidates"][representative]["seed"],
                    "representative_energy_hartree": "" if representative is None else f"{energy_by_index[representative]:.12f}",
                    "n_energy_success": len(eligible),
                }
            )
            if representative is not None:
                representatives.append((int(label), representative, energy_by_index[representative]))
        path_min = min((item[2] for item in representatives), default=None)
        stage1_min = record["stage1_min_energy"]
        medoid_delta_by_label = {
            label: (
                "" if energy is None or stage1_min is None
                else f"{energy - stage1_min:.12f}"
            )
            for label, energy in record["stage1_medoid_energy"].items()
        }
        selected_count = 0
        for label, index, energy in sorted(representatives, key=lambda item: item[2]):
            delta = energy - path_min if path_min is not None else math.nan
            # In staged mode the threshold was applied to medoid energies;
            # every surviving cluster contributes exactly one final minimum.
            keep = (
                True if args.energy_scope == "staged"
                else delta <= args.energy_window_hartree + 1e-12
            )
            candidate = record["candidates"][index]
            candidate_rows.append(
                {
                    "reaction_index": reaction_index,
                    "rxn_smiles": record["rxn_smiles"],
                    "candidate_index": index,
                    "source_candidate_index": record["candidates"][index].get(
                        "source_candidate_index", index
                    ),
                    "seed": candidate["seed"],
                    "cluster": label,
                    "cluster_size": int(np.sum(labels == label)),
                    "energy_hartree": f"{energy:.12f}",
                    "delta_energy_hartree": f"{delta:.12f}",
                    "medoid_delta_energy_hartree": medoid_delta_by_label.get(label, ""),
                    "selected": int(keep),
                    "source_xyz": str(candidate["path"].resolve()),
                }
            )
            if keep:
                selected_count += 1
                destination = selected_dir / (
                    f"rxn_{reaction_index:04d}__cluster_{label:03d}__seed_{candidate['seed']:03d}.xyz"
                )
                write_xyz(
                    destination,
                    record["numbers"],
                    candidate["coords"],
                    f"rxn_smiles={record['rxn_smiles']}",
                )
        reaction_rows.append(
            {
                "reaction_index": reaction_index,
                "reaction_key": record["reaction_key"],
                "rxn_smiles": record["rxn_smiles"],
                "n_candidates": len(record["candidates"]),
                "n_raw_candidates": record["n_raw_candidates"],
                "n_stereo_valid_candidates": record["n_stereo_valid_candidates"],
                "n_stereo_rejected_candidates": record["n_stereo_rejected_candidates"],
                "charge": record["charge"] if args.charge is None else args.charge,
                "uhf": "auto" if args.uhf is None else args.uhf,
                "n_clusters": len(np.unique(labels)),
                "n_clusters_after_medoid_filter": len(record["stage1_passed_labels"]),
                "n_energy_success": len(energy_by_index),
                "n_stage1_energy_success": sum(
                    energy is not None for energy in record["stage1_medoid_energy"].values()
                ),
                "n_stage2_energy_success": (
                    len(energy_by_index)
                    if args.energy_scope == "staged"
                    else ""
                ),
                "stage1_path_min_energy_hartree": (
                    "" if stage1_min is None else f"{stage1_min:.12f}"
                ),
                "path_min_energy_hartree": "" if path_min is None else f"{path_min:.12f}",
                "n_selected": selected_count,
            }
        )

    def write_rows(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_rows(args.output / "candidate_energy_ranking.csv", candidate_rows)
    write_rows(args.output / "cluster_summary.csv", cluster_rows)
    write_rows(args.output / "reaction_summary.csv", reaction_rows)
    write_rows(args.output / "stereo_validation.csv", stereo_rows)
    summary = {
        "candidates": str(args.candidates.resolve()),
        "reactions": None if args.reactions is None else str(args.reactions.resolve()),
        "n_reactions": len(all_records),
        "n_reactions_with_stereo_candidates": len(records),
        "n_candidates": sum(len(record["candidates"]) for record in records),
        "n_raw_candidates": sum(record["n_raw_candidates"] for record in all_records),
        "stereo_filter": stereo_stats,
        "cluster_atoms": args.cluster_atoms,
        "cluster_method": "complete-linkage Kabsch RMSD",
        "cluster_threshold_angstrom": args.cluster_threshold,
        "energy_method": "g-xTB",
        "energy_backend": "official g-xtb xtb executable with --gxtb",
        "gxtb_binary": str(binary),
        "energy_scope": args.energy_scope,
        "energy_window_hartree": args.energy_window_hartree,
        "energy_window_kcal_mol": args.energy_window_hartree * HARTREE_TO_KCAL_MOL,
        "stereo_filter_stage": (
            "strict mapped tetrahedral/E-Z sign filter before clustering"
            if args.stereo_filter
            else "disabled"
        ),
        "energy_stages": {
            "staged": [
                "cluster stereo-valid candidates by Kabsch RMSD"
                if args.stereo_filter
                else "cluster all candidates by Kabsch RMSD",
                "g-xTB medoid single point per cluster",
                "filter clusters by medoid energy window per reaction",
                "g-xTB single point for all members of retained clusters",
                "select lowest-energy member per retained cluster",
            ],
            "medoids": [
                "cluster all candidates by Kabsch RMSD",
                "g-xTB medoid single point per cluster",
                "select lowest-energy medoid per cluster",
            ],
            "all": [
                "cluster all candidates by Kabsch RMSD",
                "g-xTB single point for every candidate",
                "select lowest-energy member per cluster",
            ],
        }[args.energy_scope],
        "charge": "inferred per reaction" if args.charge is None else args.charge,
        "uhf": "g-xTB automatic" if args.uhf is None else args.uhf,
        "n_energy_success": sum(
            successful_energy(energy_cache.get(key)) is not None
            for key in current_energy_keys
        ),
        "outputs": {
            "energy_results": str(energy_path.resolve()),
            "candidate_energy_ranking": str((args.output / "candidate_energy_ranking.csv").resolve()),
            "cluster_summary": str((args.output / "cluster_summary.csv").resolve()),
            "reaction_summary": str((args.output / "reaction_summary.csv").resolve()),
            "stereo_validation": str((args.output / "stereo_validation.csv").resolve()),
            "selected_xyz": str(selected_dir.resolve()),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactions", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate", action="store_true", help="run TS generation before evaluation")
    parser.add_argument("--generation-strategy", choices=("hybrid", "global"), default="hybrid")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--start-reaction", type=int, default=0)
    parser.add_argument("--end-reaction", type=int)
    parser.add_argument(
        "--max-reactions", type=int,
        help="limit evaluation to at most this many reactions from --start-reaction",
    )
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-atoms-per-batch", type=int, default=18000)
    parser.add_argument("--cluster-threshold", type=float, default=0.5)
    parser.add_argument("--cluster-atoms", choices=("heavy", "all"), default="heavy")
    stereo_group = parser.add_mutually_exclusive_group()
    stereo_group.add_argument(
        "--stereo-filter",
        dest="stereo_filter",
        action="store_true",
        help="strict mapped tetrahedral/E-Z sign filter before clustering (default)",
    )
    stereo_group.add_argument(
        "--no-stereo-filter",
        dest="stereo_filter",
        action="store_false",
        help="disable the strict stereo filter for an explicit comparison run",
    )
    parser.set_defaults(stereo_filter=True)
    parser.add_argument(
        "--energy-scope",
        choices=("staged", "medoids", "all"),
        default="staged",
        help=(
            "staged: medoid filter then all members of retained clusters (default); "
            "medoids: medoid-only ranking; all: exhaustive ranking"
        ),
    )
    parser.add_argument("--energy-window-hartree", type=float, default=0.01)
    parser.add_argument("--gxtb-binary", type=Path)
    parser.add_argument(
        "--energy-workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="parallel g-xTB processes; each process is pinned to --omp-threads cores",
    )
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--energy-timeout", type=float, default=300.0)
    parser.add_argument("--charge", type=int, help="override charge inferred from reaction SMILES")
    parser.add_argument("--uhf", type=int, help="explicit unpaired-electron count; omitted by default")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.generate:
        if args.config is None or args.ckpt is None:
            raise SystemExit("--generate requires --config and --ckpt")
        run_generation(args)
    evaluate(args)


if __name__ == "__main__":
    main()
