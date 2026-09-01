#!/usr/bin/env python3
"""Evaluate conformer clustering and GFN1-xTB ranking of RitS candidates."""

import argparse
import csv
import json
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
import torch
from omegaconf import OmegaConf
from rdkit import Chem
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from megalodon.data.molecule_dataset import MoleculeDataset
from megalodon.data.ts_batch_preprocessor import TsBatchPreProcessor
from megalodon.interpolant.ot import rigid_alignment
from megalodon.models.module import Graph3DInterpolantModel


ENERGY_RE = re.compile(r"TOTAL ENERGY\s+([-+0-9.Ee]+)\s+Eh")
GRADIENT_RE = re.compile(r"GRADIENT NORM\s+([-+0-9.Ee]+)\s+Eh")


def scalar(value):
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().reshape(-1).tolist()
        return values[0] if values else ""
    return value


def path_key(sample, index):
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


def seeded_prior(node_count, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    coords = torch.randn((node_count, 3), generator=generator, device=device)
    return coords - coords.mean(dim=0, keepdim=True)


def kabsch_rmsd(predicted, reference, mask=None):
    if mask is not None:
        predicted = predicted[mask]
        reference = reference[mask]
    aligned = rigid_alignment(predicted.float(), reference.float())
    return torch.sqrt(
        torch.mean(torch.sum((aligned - reference.float()) ** 2, dim=-1))
    ).item()


def pairwise_kabsch_rmsd(coords, mask):
    """Return an NxN Kabsch RMSD matrix using batched SVD."""
    xyz = coords[:, mask].double()
    xyz = xyz - xyz.mean(dim=1, keepdim=True)
    left, right = torch.triu_indices(len(xyz), len(xyz), offset=1)
    p = xyz[left]
    q = xyz[right]
    covariance = p.transpose(1, 2) @ q
    u, _, vh = torch.linalg.svd(covariance)
    correction = torch.ones((len(left), 3), dtype=xyz.dtype)
    correction[:, -1] = torch.sign(torch.linalg.det(u @ vh))
    rotation = (u * correction.unsqueeze(1)) @ vh
    aligned = p @ rotation
    values = torch.sqrt(torch.mean(torch.sum((aligned - q) ** 2, dim=-1), dim=-1))
    matrix = torch.zeros((len(xyz), len(xyz)), dtype=torch.float64)
    matrix[left, right] = values
    matrix[right, left] = values
    return matrix.numpy()


def cluster_candidates(distance_matrix, threshold):
    if len(distance_matrix) == 1:
        return np.ones(1, dtype=np.int64)
    condensed = squareform(distance_matrix, checks=False)
    tree = linkage(condensed, method="complete")
    return fcluster(tree, t=threshold, criterion="distance").astype(np.int64)


def metric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"n": 0, "mean": None}
    return {
        "n": int(len(finite)),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(finite.max()),
        "fraction_le_0_5": float(np.mean(finite <= 0.5)),
        "fraction_le_1_0": float(np.mean(finite <= 1.0)),
        "fraction_le_2_0": float(np.mean(finite <= 2.0)),
    }


def write_xyz(path, numbers, coords, charge, comment=None):
    periodic_table = Chem.GetPeriodicTable()
    lines = [str(len(numbers)), comment or f"charge {charge} multiplicity 1"]
    for number, (x, y, z) in zip(numbers, coords):
        symbol = periodic_table.GetElementSymbol(int(number))
        lines.append(f"{symbol:<3s} {x: .10f} {y: .10f} {z: .10f}")
    path.write_text("\n".join(lines) + "\n")


def run_xtb(task, xtb_binary, work_root, timeout, omp_threads):
    path, candidate_index, numbers, coords, charge = task
    started = time.perf_counter()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp_threads)
    env["MKL_NUM_THREADS"] = str(omp_threads)
    env["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    with tempfile.TemporaryDirectory(dir=work_root) as tmpdir:
        xyz_path = Path(tmpdir) / "candidate.xyz"
        write_xyz(xyz_path, numbers, coords, charge)
        command = [
            str(xtb_binary),
            xyz_path.name,
            "--gfn",
            "1",
            "--sp",
            "--chrg",
            str(charge),
            "--uhf",
            "0",
        ]
        try:
            result = subprocess.run(
                command,
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            output = result.stdout + "\n" + result.stderr
            energy_matches = ENERGY_RE.findall(output)
            gradient_matches = GRADIENT_RE.findall(output)
            normal = "normal termination of xtb" in output
            success = result.returncode == 0 and normal and bool(energy_matches)
            error = "" if success else output[-1000:].replace("\n", " ")
            energy = float(energy_matches[-1]) if success else None
            gradient = float(gradient_matches[-1]) if gradient_matches else None
        except subprocess.TimeoutExpired:
            success = False
            energy = None
            gradient = None
            error = f"timeout after {timeout}s"
    return {
        "path": path,
        "candidate_index": candidate_index,
        "success": int(success),
        "energy_hartree": energy,
        "gradient_norm_hartree_per_bohr": gradient,
        "wall_seconds": time.perf_counter() - started,
        "error": error,
    }


def load_xtb_rows(csv_path):
    rows = {}
    if not csv_path.exists():
        return rows
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["path"], int(row["candidate_index"]))
            rows[key] = {
                "path": row["path"],
                "candidate_index": int(row["candidate_index"]),
                "success": int(row["success"]),
                "energy_hartree": (
                    float(row["energy_hartree"]) if row["energy_hartree"] else None
                ),
                "gradient_norm_hartree_per_bohr": (
                    float(row["gradient_norm_hartree_per_bohr"])
                    if row["gradient_norm_hartree_per_bohr"]
                    else None
                ),
                "wall_seconds": float(row["wall_seconds"]),
                "error": row["error"],
            }
    return rows


def append_xtb_rows(csv_path, rows):
    fields = [
        "path",
        "candidate_index",
        "success",
        "energy_hartree",
        "gradient_norm_hartree_per_bohr",
        "wall_seconds",
        "error",
    ]
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def generate_cache(args, cfg, cache_path):
    dataset_root = args.dataset_root or Path(cfg.data.dataset_root)
    processed_folder = args.processed_folder or cfg.data.processed_folder
    dataset = MoleculeDataset(
        root=str(dataset_root), processed_folder=processed_folder, split=args.split
    )

    grouped = defaultdict(list)
    representatives = {}
    for index, sample in enumerate(dataset):
        key = path_key(sample, index)
        grouped[key].append(
            {
                "dataset_index": index,
                "conf_id": str(scalar(getattr(sample, "conf_id", "")) or ""),
                "coords": sample.ts_coord.detach().cpu(),
                "numbers": sample.numbers.detach().cpu(),
            }
        )
        representatives.setdefault(key, sample.clone())

    all_keys = list(grouped)
    eligible = [
        index
        for index, key in enumerate(all_keys)
        if len(grouped[key]) >= args.min_references
        and (
            args.max_references is None
            or len(grouped[key]) <= args.max_references
        )
    ]
    if args.max_paths and len(eligible) > args.max_paths:
        rng = np.random.default_rng(args.subset_seed)
        eligible = sorted(rng.choice(eligible, args.max_paths, replace=False).tolist())

    candidate_graphs = []
    metadata = []
    for global_path_index in eligible:
        key = all_keys[global_path_index]
        representative = representatives[key]
        for candidate_index in range(args.num_candidates):
            candidate_graphs.append(representative.clone())
            metadata.append((global_path_index, key, candidate_index))

    batch_preprocessor = TsBatchPreProcessor(
        aug_rotations=False,
        scale_coords=cfg.data.get("scale_coords", 1.0),
        ts_ratio=1.0,
    )
    model = Graph3DInterpolantModel.load_from_checkpoint(
        args.ckpt,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=batch_preprocessor,
        strict=True,
    ).to(args.device).eval()

    generated = {all_keys[index]: [None] * args.num_candidates for index in eligible}
    loader = DataLoader(
        candidate_graphs, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    cursor = 0
    coord_scale = float(cfg.data.get("scale_coords", 1.0))
    for batch in tqdm(loader, desc=f"{args.split} candidate sampling"):
        graph_count = int(batch.num_graphs)
        node_counts = torch.bincount(batch.batch).tolist()
        batch_metadata = metadata[cursor : cursor + graph_count]
        priors = []
        for node_count, (global_path_index, _, candidate_index) in zip(
            node_counts, batch_metadata
        ):
            prior_seed = args.seed + global_path_index * 1_000_003 + candidate_index
            priors.append(seeded_prior(int(node_count), prior_seed, args.device))
        batch = batch.to(args.device)
        with torch.inference_mode():
            sampled = model.sample(
                batch=batch,
                timesteps=args.num_steps,
                pre_format=True,
                initial_priors={"ts_coord": torch.cat(priors, dim=0)},
            )
        predictions = torch.split(sampled["ts_coord"] * coord_scale, node_counts)
        for prediction, (_, key, candidate_index) in zip(
            predictions, batch_metadata
        ):
            generated[key][candidate_index] = prediction.detach().cpu()
        cursor += graph_count

    records = []
    for global_path_index in eligible:
        key = all_keys[global_path_index]
        representative = representatives[key]
        unique_charges = torch.unique(representative.charges.detach().cpu())
        if len(unique_charges) != 1:
            raise ValueError(f"Expected one repeated molecular charge for {key}")
        records.append(
            {
                "path": key,
                "global_path_index": global_path_index,
                "numbers": representative.numbers.detach().cpu(),
                "charge": int(unique_charges.item()),
                "references": grouped[key],
                "candidates": torch.stack(generated[key]),
            }
        )
    cache = {
        "checkpoint": str(args.ckpt.resolve()),
        "config": str(args.config.resolve()),
        "split": args.split,
        "num_candidates": args.num_candidates,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "subset_seed": args.subset_seed,
        "min_references": args.min_references,
        "max_references": args.max_references,
        "records": records,
    }
    torch.save(cache, cache_path)
    return cache


def cluster_medoids(labels, distance_matrix):
    """Choose a geometry-only medoid before any energy calculations."""
    medoids = []
    sizes = {}
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        intra_cluster_mean = distance_matrix[np.ix_(members, members)].mean(axis=1)
        medoid = int(members[np.argmin(intra_cluster_mean)])
        medoids.append(medoid)
        sizes[medoid] = int(len(members))
    return np.asarray(medoids, dtype=int), sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--processed_folder", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--num_candidates", type=int, default=100)
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_paths", type=int, default=50)
    parser.add_argument("--min_references", type=int, default=2)
    parser.add_argument("--max_references", type=int, default=None)
    parser.add_argument("--subset_seed", type=int, default=20260814)
    parser.add_argument("--cluster_threshold", type=float, default=0.5)
    parser.add_argument(
        "--cluster_atoms", choices=("heavy", "all"), default="heavy"
    )
    parser.add_argument("--xtb_binary", type=Path, required=True)
    parser.add_argument("--xtb_workers", type=int, default=16)
    parser.add_argument("--xtb_omp_threads", type=int, default=1)
    parser.add_argument("--xtb_timeout", type=float, default=120.0)
    parser.add_argument(
        "--xtb_scope",
        choices=("medoids", "all_candidates"),
        default="medoids",
        help="Run xTB for geometry medoids or every generated candidate.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument(
        "--export_xyz",
        action="store_true",
        help="Write ranked candidate XYZ files under output_dir/xyz.",
    )
    parser.add_argument(
        "--export_strategies",
        nargs="+",
        default=["cluster_medoid_xtb_energy"],
        help="Ranking strategies to export when --export_xyz is set.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "candidates.pt"
    cfg = OmegaConf.load(args.config)
    if cache_path.exists() and not args.regenerate:
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        expected = {
            "checkpoint": str(args.ckpt.resolve()),
            "config": str(args.config.resolve()),
            "split": args.split,
            "num_candidates": args.num_candidates,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "subset_seed": args.subset_seed,
            "min_references": args.min_references,
            "max_references": args.max_references,
        }
        mismatches = {
            key: (cache.get(key), value)
            for key, value in expected.items()
            if cache.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Candidate cache argument mismatch: {mismatches}")
        print(f"Loaded candidate cache: {cache_path}")
    else:
        cache = generate_cache(args, cfg, cache_path)

    records = cache["records"]
    clustered = {}
    cluster_label = f"{args.cluster_atoms}-atom RMSD clustering"
    for record in tqdm(records, desc=cluster_label):
        cluster_mask = (
            record["numbers"] > 1
            if args.cluster_atoms == "heavy"
            else record["numbers"] > 0
        )
        distance_matrix = pairwise_kabsch_rmsd(record["candidates"], cluster_mask)
        labels = cluster_candidates(distance_matrix, args.cluster_threshold)
        medoids, cluster_sizes = cluster_medoids(labels, distance_matrix)
        clustered[record["path"]] = {
            "distance_matrix": distance_matrix,
            "labels": labels,
            "medoids": medoids,
            "cluster_sizes": cluster_sizes,
        }

    work_root = args.output_dir / "xtb_work"
    work_root.mkdir(exist_ok=True)
    xtb_csv = args.output_dir / "xtb_results.csv"
    xtb_rows = load_xtb_rows(xtb_csv)
    tasks = []
    for record in records:
        numbers = record["numbers"].tolist()
        medoids = clustered[record["path"]]["medoids"]
        energy_indices = (
            range(len(record["candidates"]))
            if args.xtb_scope == "all_candidates"
            else medoids
        )
        for candidate_index in energy_indices:
            coords = record["candidates"][candidate_index]
            key = (record["path"], candidate_index)
            if key not in xtb_rows:
                tasks.append(
                    (
                        record["path"],
                        candidate_index,
                        numbers,
                        coords.tolist(),
                        record["charge"],
                    )
                )

    if tasks:
        pending_rows = []
        with ThreadPoolExecutor(max_workers=args.xtb_workers) as executor:
            futures = [
                executor.submit(
                    run_xtb,
                    task,
                    args.xtb_binary.resolve(),
                    work_root.resolve(),
                    args.xtb_timeout,
                    args.xtb_omp_threads,
                )
                for task in tasks
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="GFN1-xTB single points"
            ):
                row = future.result()
                xtb_rows[(row["path"], row["candidate_index"])] = row
                pending_rows.append(row)
                if len(pending_rows) >= 100:
                    append_xtb_rows(xtb_csv, pending_rows)
                    pending_rows.clear()
        if pending_rows:
            append_xtb_rows(xtb_csv, pending_rows)
    shutil.rmtree(work_root, ignore_errors=True)

    metric_values = defaultdict(lambda: defaultdict(list))
    candidate_rows = []
    path_rows = []
    xyz_exports = defaultdict(int)
    xyz_rows = []
    correlation_all = []
    correlation_heavy = []
    xtb_wall_times = []
    xtb_successes = 0
    xtb_total = 0

    for record in tqdm(records, desc="Ranking analysis"):
        candidates = record["candidates"]
        references = record["references"]
        numbers = record["numbers"]
        heavy_mask = numbers > 1
        all_matrix = np.empty((len(candidates), len(references)), dtype=np.float64)
        heavy_matrix = np.empty_like(all_matrix)
        for candidate_index, candidate in enumerate(candidates):
            for reference_index, reference in enumerate(references):
                if not torch.equal(numbers, reference["numbers"]):
                    raise ValueError(f"Mapped atom order differs within {record['path']}")
                all_matrix[candidate_index, reference_index] = kabsch_rmsd(
                    candidate, reference["coords"]
                )
                heavy_matrix[candidate_index, reference_index] = kabsch_rmsd(
                    candidate, reference["coords"], heavy_mask
                )
        nearest_all = all_matrix.min(axis=1)
        nearest_heavy = heavy_matrix.min(axis=1)
        cluster_info = clustered[record["path"]]
        labels = cluster_info["labels"]
        medoids = cluster_info["medoids"]
        cluster_sizes = cluster_info["cluster_sizes"]

        energies = np.full(len(candidates), np.nan)
        gradients = np.full(len(candidates), np.nan)
        energy_indices = (
            range(len(candidates))
            if args.xtb_scope == "all_candidates"
            else medoids
        )
        for candidate_index in energy_indices:
            row = xtb_rows[(record["path"], candidate_index)]
            xtb_total += 1
            xtb_successes += row["success"]
            xtb_wall_times.append(row["wall_seconds"])
            if row["energy_hartree"] is not None:
                energies[candidate_index] = row["energy_hartree"]
            if row["gradient_norm_hartree_per_bohr"] is not None:
                gradients[candidate_index] = row["gradient_norm_hartree_per_bohr"]

        successful = np.flatnonzero(np.isfinite(energies))
        cluster_energy_min = []
        for cluster_label in np.unique(labels):
            members = np.flatnonzero(labels == cluster_label)
            available = members[np.isfinite(energies[members])]
            if len(available):
                cluster_energy_min.append(
                    int(available[np.argmin(energies[available])])
                )
        cluster_energy_min = np.asarray(
            sorted(cluster_energy_min, key=lambda index: energies[index]),
            dtype=int,
        )
        if len(successful) >= 3:
            correlation_all.append(
                float(spearmanr(energies[successful], nearest_all[successful]).statistic)
            )
            correlation_heavy.append(
                float(
                    spearmanr(energies[successful], nearest_heavy[successful]).statistic
                )
            )

        medoid_generation_order = np.sort(medoids)
        rankings = {
            "generation_order": np.arange(len(candidates), dtype=int),
            "oracle_all_candidates_all_atom": np.argsort(nearest_all),
            "oracle_all_candidates_heavy_atom": np.argsort(nearest_heavy),
            "cluster_medoid_generation_order": medoid_generation_order,
            "cluster_medoid_oracle_all_atom": medoids[np.argsort(nearest_all[medoids])],
            "cluster_medoid_oracle_heavy_atom": medoids[
                np.argsort(nearest_heavy[medoids])
            ],
            "cluster_medoid_xtb_energy": successful[np.argsort(energies[successful])],
            "cluster_population_then_energy": np.asarray(
                sorted(
                    successful.tolist(),
                    key=lambda index: (
                        -int(np.sum(labels == labels[index])),
                        energies[index],
                    ),
                ),
                dtype=int,
            ),
            "cluster_energy_min": cluster_energy_min,
        }

        path_row = {
            "path": record["path"],
            "global_path_index": record["global_path_index"],
            "n_atoms": len(numbers),
            "n_heavy_atoms": int(heavy_mask.sum()),
            "n_references": len(references),
            "n_clusters": len(medoids),
            "xtb_successes": len(successful),
            "energy_rmsd_spearman_all_atom": (
                float(spearmanr(energies[successful], nearest_all[successful]).statistic)
                if len(successful) >= 3
                else None
            ),
            "energy_rmsd_spearman_heavy_atom": (
                float(
                    spearmanr(energies[successful], nearest_heavy[successful]).statistic
                )
                if len(successful) >= 3
                else None
            ),
        }
        ranks_by_strategy = {
            name: {int(candidate): rank + 1 for rank, candidate in enumerate(ranking)}
            for name, ranking in rankings.items()
        }
        if args.export_xyz:
            export_root = args.output_dir / "xyz"
            safe_path = re.sub(r"[^A-Za-z0-9_.-]+", "_", record["path"])
            for strategy in args.export_strategies:
                if strategy not in rankings:
                    raise ValueError(
                        f"Unknown --export_strategies value: {strategy}; "
                        f"available: {sorted(rankings)}"
                    )
                strategy_dir = export_root / strategy
                strategy_dir.mkdir(parents=True, exist_ok=True)
                for rank, candidate_index in enumerate(
                    rankings[strategy][: max(args.top_k)], start=1
                ):
                    candidate_index = int(candidate_index)
                    energy = energies[candidate_index]
                    energy_text = (
                        f"{energy:.12f}" if np.isfinite(energy) else "nan"
                    )
                    comment = (
                        f"path={record['path']} strategy={strategy} rank={rank} "
                        f"candidate_index={candidate_index} cluster={labels[candidate_index]} "
                        f"cluster_size={int(np.sum(labels == labels[candidate_index]))} "
                        f"charge={record['charge']} xtb_energy_hartree={energy_text} "
                        f"nearest_reference_all_atom_rmsd={nearest_all[candidate_index]:.6f} "
                        f"nearest_reference_heavy_atom_rmsd={nearest_heavy[candidate_index]:.6f}"
                    )
                    xyz_path = strategy_dir / (
                        f"path_{record['global_path_index']:04d}_{safe_path}"
                        f"__rank_{rank:02d}__candidate_{candidate_index:03d}.xyz"
                    )
                    write_xyz(
                        xyz_path,
                        numbers,
                        candidates[candidate_index].tolist(),
                        record["charge"],
                        comment=comment,
                    )
                    xyz_exports[strategy] += 1
                    xyz_rows.append(
                        {
                            "path": record["path"],
                            "global_path_index": record["global_path_index"],
                            "strategy": strategy,
                            "rank": rank,
                            "candidate_index": candidate_index,
                            "cluster": int(labels[candidate_index]),
                            "cluster_size": int(np.sum(labels == labels[candidate_index])),
                            "charge": record["charge"],
                            "xtb_energy_hartree": energy_text,
                            "nearest_reference_all_atom_rmsd": (
                                f"{nearest_all[candidate_index]:.6f}"
                            ),
                            "nearest_reference_heavy_atom_rmsd": (
                                f"{nearest_heavy[candidate_index]:.6f}"
                            ),
                            "xyz_path": str(xyz_path.resolve()),
                        }
                    )
        for strategy, ranking in rankings.items():
            for k in sorted(set(args.top_k)):
                selected = ranking[:k]
                if not len(selected):
                    all_score = np.nan
                    heavy_score = np.nan
                    coverage_all = np.nan
                    coverage_heavy = np.nan
                else:
                    all_score = float(nearest_all[selected].min())
                    heavy_score = float(nearest_heavy[selected].min())
                    coverage_all = float(all_matrix[selected].min(axis=0).mean())
                    coverage_heavy = float(heavy_matrix[selected].min(axis=0).mean())
                values = metric_values[strategy]
                values[f"top{k}_best_all"].append(all_score)
                values[f"top{k}_best_heavy"].append(heavy_score)
                values[f"top{k}_coverage_all"].append(coverage_all)
                values[f"top{k}_coverage_heavy"].append(coverage_heavy)
                path_row[f"{strategy}_top{k}_best_all_atom_rmsd"] = all_score
                path_row[f"{strategy}_top{k}_best_heavy_atom_rmsd"] = heavy_score
        path_rows.append(path_row)

        for candidate_index in range(len(candidates)):
            row = {
                "path": record["path"],
                "candidate_index": candidate_index,
                "nearest_reference_all_atom_rmsd": nearest_all[candidate_index],
                "nearest_reference_heavy_atom_rmsd": nearest_heavy[candidate_index],
                "xtb_success": int(np.isfinite(energies[candidate_index])),
                "energy_hartree": energies[candidate_index],
                "gradient_norm_hartree_per_bohr": gradients[candidate_index],
                "cluster": int(labels[candidate_index]),
                "is_cluster_medoid": int(candidate_index in cluster_sizes),
                "cluster_size": int(np.sum(labels == labels[candidate_index])),
            }
            for strategy, rank_map in ranks_by_strategy.items():
                row[f"{strategy}_rank"] = rank_map.get(candidate_index, "")
            candidate_rows.append(row)

    per_candidate_csv = args.output_dir / "per_candidate.csv"
    per_path_csv = args.output_dir / "per_path.csv"
    with per_candidate_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    with per_path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(path_rows[0]))
        writer.writeheader()
        writer.writerows(path_rows)
    xyz_manifest = args.output_dir / "xyz_manifest.csv"
    if xyz_rows:
        with xyz_manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(xyz_rows[0]))
            writer.writeheader()
            writer.writerows(xyz_rows)

    strategy_summary = {}
    for strategy, values in metric_values.items():
        strategy_summary[strategy] = {
            metric: metric_summary(scores) for metric, scores in values.items()
        }
    cluster_counts = [row["n_clusters"] for row in path_rows]
    summary = {
        "checkpoint": cache["checkpoint"],
        "config": cache["config"],
        "split": cache["split"],
        "n_paths": len(records),
        "n_references": int(sum(len(record["references"]) for record in records)),
        "num_candidates_per_path": cache["num_candidates"],
        "num_steps": cache["num_steps"],
        "base_seed": cache["seed"],
        "subset_seed": cache["subset_seed"],
        "min_references": cache["min_references"],
        "max_references": cache.get("max_references"),
        "cluster_method": (
            f"complete-linkage {args.cluster_atoms}-atom Kabsch RMSD"
        ),
        "cluster_threshold_angstrom": args.cluster_threshold,
        "cluster_counts": metric_summary(cluster_counts),
        "xtb_screened_fraction_of_generated": float(
            sum(cluster_counts) / (len(records) * cache["num_candidates"])
        ),
        "xtb": {
            "binary": str(args.xtb_binary.resolve()),
            "method": "GFN1-xTB single point",
            "successes": xtb_successes,
            "total": xtb_total,
            "success_fraction": xtb_successes / xtb_total,
            "wall_seconds_per_candidate": metric_summary(xtb_wall_times),
        },
        "within_path_spearman_energy_vs_nearest_reference_rmsd": {
            "all_atom": metric_summary(correlation_all),
            "heavy_atom": metric_summary(correlation_heavy),
        },
        "strategies": strategy_summary,
        "xyz_exports": {
            "enabled": bool(args.export_xyz),
            "root": str((args.output_dir / "xyz").resolve())
            if args.export_xyz
            else None,
            "files_by_strategy": dict(xyz_exports),
            "manifest": str(xyz_manifest.resolve()) if xyz_rows else None,
            "comment_fields": [
                "path",
                "strategy",
                "rank",
                "candidate_index",
                "cluster",
                "cluster_size",
                "charge",
                "xtb_energy_hartree",
                "nearest_reference_all_atom_rmsd",
                "nearest_reference_heavy_atom_rmsd",
            ],
        },
        "per_candidate_csv": str(per_candidate_csv.resolve()),
        "per_path_csv": str(per_path_csv.resolve()),
        "xtb_results_csv": str(xtb_csv.resolve()),
        "candidate_cache": str(cache_path.resolve()),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
