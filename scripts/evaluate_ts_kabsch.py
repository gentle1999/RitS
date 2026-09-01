#!/usr/bin/env python3
"""Evaluate sampled transition states with atom-ordered Kabsch RMSD."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
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


def kabsch_rmsd(predicted: torch.Tensor, reference: torch.Tensor) -> float:
    """Return proper-rotation Kabsch RMSD for two atom-ordered structures."""
    if predicted.shape != reference.shape:
        raise ValueError(
            f"coordinate shapes differ: {tuple(predicted.shape)} != {tuple(reference.shape)}"
        )
    aligned = rigid_alignment(predicted.float(), reference.float())
    return torch.sqrt(torch.mean(torch.sum((aligned - reference.float()) ** 2, dim=-1))).item()


def seeded_coordinate_prior(node_counts, start_index, seed, device):
    """Build graph-independent priors so batching does not change evaluation."""
    priors = []
    for offset, n_atoms in enumerate(node_counts):
        generator = torch.Generator(device=device).manual_seed(seed + start_index + offset)
        coords = torch.randn((int(n_atoms), 3), generator=generator, device=device)
        priors.append(coords - coords.mean(dim=0, keepdim=True))
    return torch.cat(priors, dim=0)


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
        "fraction_le_0_5": float(np.mean(values <= 0.5)),
        "fraction_le_1_0": float(np.mean(values <= 1.0)),
        "fraction_le_2_0": float(np.mean(values <= 2.0)),
    }


def as_list(value, count):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * count


def reaction_key(row):
    if row["reaction_id"] != "":
        return f"reaction_id:{row['reaction_id']}"
    if row["rxn_index"] != "":
        return f"rxn_index:{row['rxn_index']}"
    dpa_key = (row["ene_id"], row["diene_id"], row["prod_id"])
    if any(value != "" for value in dpa_key):
        return "dpa:" + "_".join(str(value) for value in dpa_key)
    return f"sample:{row['dataset_index']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--processed_folder", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    if args.num_steps <= 0:
        parser.error("--num_steps must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")

    cfg = OmegaConf.load(args.config)
    dataset_root = args.dataset_root or Path(cfg.data.dataset_root)
    processed_folder = args.processed_folder or cfg.data.processed_folder
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.ckpt.parent.parent / (
            f"{args.split}_kabsch_{args.num_steps}step_seed{args.seed}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

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

    dataset = MoleculeDataset(
        root=str(dataset_root),
        processed_folder=processed_folder,
        split=args.split,
    )
    n_samples = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    subset = dataset[:n_samples]
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    coord_scale = float(cfg.data.get("scale_coords", 1.0))

    rows = []
    structures = []
    sample_index = 0
    for batch in tqdm(loader, desc=f"{args.split} sampling"):
        graph_count = int(batch.num_graphs)
        node_counts = torch.bincount(batch.batch).tolist()
        references = list(torch.split(batch.ts_coord.clone(), node_counts))
        numbers = list(torch.split(batch.numbers.clone(), node_counts))
        ids = as_list(batch.id, graph_count)
        reaction_ids = as_list(getattr(batch, "reaction_id", ""), graph_count)
        # PyG increments attributes whose names contain "index" while batching,
        # so read reaction metadata from the original, unbatched samples.
        source_samples = [subset[sample_index + i] for i in range(graph_count)]
        rxn_indices = [
            as_list(getattr(sample, "rxn_index", ""), 1)[0]
            for sample in source_samples
        ]
        augmented = [
            as_list(getattr(sample, "augmented", ""), 1)[0]
            for sample in source_samples
        ]
        ene_ids = as_list(getattr(batch, "ene_id", ""), graph_count)
        diene_ids = as_list(getattr(batch, "diene_id", ""), graph_count)
        prod_ids = as_list(getattr(batch, "prod_id", ""), graph_count)
        conf_ids = as_list(getattr(batch, "conf_id", ""), graph_count)

        batch = batch.to(args.device)
        initial_priors = {
            "ts_coord": seeded_coordinate_prior(
                node_counts, sample_index, args.seed, args.device
            )
        }
        with torch.inference_mode():
            sampled = model.sample(
                batch=batch,
                timesteps=args.num_steps,
                pre_format=True,
                initial_priors=initial_priors,
            )
        predictions = list(torch.split(sampled["ts_coord"] * coord_scale, node_counts))

        for local_idx, (predicted, reference, atomic_numbers) in enumerate(
            zip(predictions, references, numbers)
        ):
            reference = reference.to(predicted.device)
            atomic_numbers = atomic_numbers.to(predicted.device)
            all_atom_rmsd = kabsch_rmsd(predicted, reference)
            heavy_mask = atomic_numbers > 1
            heavy_atom_rmsd = (
                kabsch_rmsd(predicted[heavy_mask], reference[heavy_mask])
                if int(heavy_mask.sum()) >= 3
                else float("nan")
            )
            rows.append(
                {
                    "dataset_index": sample_index + local_idx,
                    "id": ids[local_idx],
                    "reaction_id": reaction_ids[local_idx],
                    "rxn_index": rxn_indices[local_idx],
                    "augmented": augmented[local_idx],
                    "ene_id": ene_ids[local_idx],
                    "diene_id": diene_ids[local_idx],
                    "prod_id": prod_ids[local_idx],
                    "conf_id": conf_ids[local_idx],
                    "n_atoms": int(node_counts[local_idx]),
                    "n_heavy_atoms": int(heavy_mask.sum()),
                    "kabsch_rmsd_angstrom": all_atom_rmsd,
                    "heavy_atom_kabsch_rmsd_angstrom": heavy_atom_rmsd,
                }
            )
            structures.append(
                {
                    "predicted": predicted.detach().cpu(),
                    "reference": reference.detach().cpu(),
                    "numbers": atomic_numbers.detach().cpu(),
                }
            )
        sample_index += graph_count

    reaction_indices = defaultdict(list)
    for index, row in enumerate(rows):
        reaction_indices[reaction_key(row)].append(index)

    reference_coverage_rmsds = []
    heavy_reference_coverage_rmsds = []
    multiconformer_reference_coverage_rmsds = []
    multiconformer_heavy_reference_coverage_rmsds = []
    generated_path_mean_rmsds = []
    heavy_generated_path_mean_rmsds = []
    reference_path_mean_rmsds = []
    heavy_reference_path_mean_rmsds = []
    multiconformer_reference_path_mean_rmsds = []
    multiconformer_heavy_reference_path_mean_rmsds = []
    for indices in reaction_indices.values():
        distance_matrix = np.empty((len(indices), len(indices)), dtype=np.float64)
        heavy_distance_matrix = np.empty(
            (len(indices), len(indices)), dtype=np.float64
        )
        for pred_idx, row_idx in enumerate(indices):
            predicted = structures[row_idx]["predicted"]
            for ref_idx, ref_row_idx in enumerate(indices):
                reference = structures[ref_row_idx]["reference"]
                distance_matrix[pred_idx, ref_idx] = kabsch_rmsd(predicted, reference)
                heavy_mask = structures[ref_row_idx]["numbers"] > 1
                heavy_distance_matrix[pred_idx, ref_idx] = kabsch_rmsd(
                    predicted[heavy_mask], reference[heavy_mask]
                )
        generated_minima = distance_matrix.min(axis=1)
        reference_minima = distance_matrix.min(axis=0)
        heavy_generated_minima = heavy_distance_matrix.min(axis=1)
        heavy_reference_minima = heavy_distance_matrix.min(axis=0)
        for row_idx, nearest_rmsd, nearest_heavy_rmsd in zip(
            indices, generated_minima, heavy_generated_minima
        ):
            rows[row_idx]["nearest_reference_kabsch_rmsd_angstrom"] = float(nearest_rmsd)
            rows[row_idx]["nearest_reference_heavy_atom_kabsch_rmsd_angstrom"] = float(
                nearest_heavy_rmsd
            )
        reference_coverage_rmsds.extend(reference_minima.tolist())
        heavy_reference_coverage_rmsds.extend(heavy_reference_minima.tolist())
        generated_path_mean_rmsds.append(float(generated_minima.mean()))
        heavy_generated_path_mean_rmsds.append(float(heavy_generated_minima.mean()))
        reference_path_mean_rmsds.append(float(reference_minima.mean()))
        heavy_reference_path_mean_rmsds.append(float(heavy_reference_minima.mean()))
        if len(indices) > 1:
            multiconformer_reference_coverage_rmsds.extend(reference_minima.tolist())
            multiconformer_heavy_reference_coverage_rmsds.extend(
                heavy_reference_minima.tolist()
            )
            multiconformer_reference_path_mean_rmsds.append(
                float(reference_minima.mean())
            )
            multiconformer_heavy_reference_path_mean_rmsds.append(
                float(heavy_reference_minima.mean())
            )

    csv_path = output_dir / "per_structure.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reaction_values = defaultdict(list)
    for row in rows:
        reaction_values[reaction_key(row)].append(row["kabsch_rmsd_angstrom"])
    reaction_means = [float(np.mean(values)) for values in reaction_values.values()]

    summary = {
        "checkpoint": str(args.ckpt.resolve()),
        "config": str(args.config.resolve()),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "n_structures": len(rows),
        "n_reactions": len(reaction_values),
        "atom_correspondence": "mapped atom order; no permutation matching",
        "units": "angstrom",
        "all_atom_kabsch_rmsd": metric_summary(
            [row["kabsch_rmsd_angstrom"] for row in rows]
        ),
        "heavy_atom_kabsch_rmsd": metric_summary(
            [row["heavy_atom_kabsch_rmsd_angstrom"] for row in rows]
        ),
        "reaction_mean_all_atom_kabsch_rmsd": metric_summary(reaction_means),
        "generated_to_nearest_reference_kabsch_rmsd": metric_summary(
            [row["nearest_reference_kabsch_rmsd_angstrom"] for row in rows]
        ),
        "generated_to_nearest_reference_heavy_atom_kabsch_rmsd": metric_summary(
            [
                row["nearest_reference_heavy_atom_kabsch_rmsd_angstrom"]
                for row in rows
            ]
        ),
        "reference_to_nearest_generated_kabsch_rmsd": metric_summary(
            reference_coverage_rmsds
        ),
        "reference_to_nearest_generated_heavy_atom_kabsch_rmsd": metric_summary(
            heavy_reference_coverage_rmsds
        ),
        "multiconformer_reference_to_nearest_generated_kabsch_rmsd": metric_summary(
            multiconformer_reference_coverage_rmsds
        ),
        "multiconformer_reference_to_nearest_generated_heavy_atom_kabsch_rmsd": metric_summary(
            multiconformer_heavy_reference_coverage_rmsds
        ),
        "path_mean_generated_to_nearest_reference_kabsch_rmsd": metric_summary(
            generated_path_mean_rmsds
        ),
        "path_mean_generated_to_nearest_reference_heavy_atom_kabsch_rmsd": metric_summary(
            heavy_generated_path_mean_rmsds
        ),
        "path_mean_reference_to_nearest_generated_kabsch_rmsd": metric_summary(
            reference_path_mean_rmsds
        ),
        "path_mean_reference_to_nearest_generated_heavy_atom_kabsch_rmsd": metric_summary(
            heavy_reference_path_mean_rmsds
        ),
        "multiconformer_path_mean_reference_to_nearest_generated_kabsch_rmsd": metric_summary(
            multiconformer_reference_path_mean_rmsds
        ),
        "multiconformer_path_mean_reference_to_nearest_generated_heavy_atom_kabsch_rmsd": metric_summary(
            multiconformer_heavy_reference_path_mean_rmsds
        ),
        "per_structure_csv": str(csv_path.resolve()),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
