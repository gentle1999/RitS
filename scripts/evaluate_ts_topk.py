#!/usr/bin/env python3
"""Evaluate fixed-size TS candidate sets against multiconformer references."""

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


def kabsch_rmsd(predicted, reference, mask=None):
    if mask is not None:
        predicted = predicted[mask]
        reference = reference[mask]
    aligned = rigid_alignment(predicted.float(), reference.float())
    return torch.sqrt(torch.mean(torch.sum((aligned - reference.float()) ** 2, dim=-1))).item()


def metric_summary(values):
    values = np.asarray(values, dtype=np.float64)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--processed_folder", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    top_k = sorted(set(args.top_k))
    if not top_k or top_k[0] <= 0 or top_k[-1] > args.num_candidates:
        parser.error("--top_k values must be between 1 and --num_candidates")

    cfg = OmegaConf.load(args.config)
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

    keys = list(grouped)
    candidate_graphs = []
    candidate_metadata = []
    for path_index, key in enumerate(keys):
        representative = representatives[key]
        for candidate_index in range(args.num_candidates):
            candidate_graphs.append(representative.clone())
            candidate_metadata.append((path_index, candidate_index))

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

    generated = [[None] * args.num_candidates for _ in keys]
    loader = DataLoader(
        candidate_graphs, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    cursor = 0
    coord_scale = float(cfg.data.get("scale_coords", 1.0))
    for batch in tqdm(loader, desc=f"{args.split} top-k sampling"):
        graph_count = int(batch.num_graphs)
        node_counts = torch.bincount(batch.batch).tolist()
        metadata = candidate_metadata[cursor : cursor + graph_count]
        priors = []
        for node_count, (path_index, candidate_index) in zip(node_counts, metadata):
            # Keep each path/candidate stream stable when num_candidates changes.
            prior_seed = args.seed + path_index * 1_000_003 + candidate_index
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
        for prediction, (path_index, candidate_index) in zip(predictions, metadata):
            generated[path_index][candidate_index] = prediction.detach().cpu()
        cursor += graph_count

    metric_values = {
        k: {
            "path_best_all": [],
            "path_best_heavy": [],
            "coverage_micro_all": [],
            "coverage_micro_heavy": [],
            "coverage_macro_all": [],
            "coverage_macro_heavy": [],
            "precision_all": [],
            "precision_heavy": [],
        }
        for k in top_k
    }
    path_rows = []
    reference_rows = []
    for path_index, key in enumerate(keys):
        references = grouped[key]
        predictions = generated[path_index]
        numbers = references[0]["numbers"]
        heavy_mask = numbers > 1
        all_matrix = np.empty(
            (args.num_candidates, len(references)), dtype=np.float64
        )
        heavy_matrix = np.empty_like(all_matrix)
        for candidate_index, prediction in enumerate(predictions):
            for reference_index, reference in enumerate(references):
                if not torch.equal(numbers, reference["numbers"]):
                    raise ValueError(f"Mapped atom order differs within {key}")
                all_matrix[candidate_index, reference_index] = kabsch_rmsd(
                    prediction, reference["coords"]
                )
                heavy_matrix[candidate_index, reference_index] = kabsch_rmsd(
                    prediction, reference["coords"], mask=heavy_mask
                )

        path_row = {"path": key, "n_references": len(references)}
        path_reference_rows = [
            {
                "path": key,
                "dataset_index": reference["dataset_index"],
                "conf_id": reference["conf_id"],
                "n_references": len(references),
            }
            for reference in references
        ]
        for k in top_k:
            all_subset = all_matrix[:k]
            heavy_subset = heavy_matrix[:k]
            coverage_all = all_subset.min(axis=0)
            coverage_heavy = heavy_subset.min(axis=0)
            precision_all = all_subset.min(axis=1)
            precision_heavy = heavy_subset.min(axis=1)
            values = metric_values[k]
            values["path_best_all"].append(float(all_subset.min()))
            values["path_best_heavy"].append(float(heavy_subset.min()))
            values["coverage_micro_all"].extend(coverage_all.tolist())
            values["coverage_micro_heavy"].extend(coverage_heavy.tolist())
            values["coverage_macro_all"].append(float(coverage_all.mean()))
            values["coverage_macro_heavy"].append(float(coverage_heavy.mean()))
            values["precision_all"].extend(precision_all.tolist())
            values["precision_heavy"].extend(precision_heavy.tolist())
            path_row[f"top{k}_best_any_reference_all_atom_rmsd"] = float(
                all_subset.min()
            )
            path_row[f"top{k}_best_any_reference_heavy_atom_rmsd"] = float(
                heavy_subset.min()
            )
            path_row[f"top{k}_reference_coverage_all_atom_rmsd"] = float(
                coverage_all.mean()
            )
            path_row[f"top{k}_reference_coverage_heavy_atom_rmsd"] = float(
                coverage_heavy.mean()
            )
            for row, all_value, heavy_value in zip(
                path_reference_rows, coverage_all, coverage_heavy
            ):
                row[f"top{k}_nearest_generated_all_atom_rmsd"] = float(all_value)
                row[f"top{k}_nearest_generated_heavy_atom_rmsd"] = float(
                    heavy_value
                )
        path_rows.append(path_row)
        reference_rows.extend(path_reference_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_path_csv = args.output_dir / "per_path.csv"
    per_reference_csv = args.output_dir / "per_reference.csv"
    with per_path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(path_rows[0]))
        writer.writeheader()
        writer.writerows(path_rows)
    with per_reference_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reference_rows[0]))
        writer.writeheader()
        writer.writerows(reference_rows)

    topk_summary = {}
    for k in top_k:
        values = metric_values[k]
        topk_summary[f"top{k}"] = {
            "oracle_path_best_to_any_reference_all_atom_rmsd": metric_summary(
                values["path_best_all"]
            ),
            "oracle_path_best_to_any_reference_heavy_atom_rmsd": metric_summary(
                values["path_best_heavy"]
            ),
            "reference_coverage_all_atom_rmsd_micro": metric_summary(
                values["coverage_micro_all"]
            ),
            "reference_coverage_heavy_atom_rmsd_micro": metric_summary(
                values["coverage_micro_heavy"]
            ),
            "reference_coverage_all_atom_rmsd_path_macro": metric_summary(
                values["coverage_macro_all"]
            ),
            "reference_coverage_heavy_atom_rmsd_path_macro": metric_summary(
                values["coverage_macro_heavy"]
            ),
            "generated_candidate_to_nearest_reference_all_atom_rmsd": metric_summary(
                values["precision_all"]
            ),
            "generated_candidate_to_nearest_reference_heavy_atom_rmsd": metric_summary(
                values["precision_heavy"]
            ),
        }

    summary = {
        "checkpoint": str(args.ckpt.resolve()),
        "config": str(args.config.resolve()),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "n_paths": len(keys),
        "n_references": len(dataset),
        "num_candidates_per_path": args.num_candidates,
        "top_k": top_k,
        "num_steps": args.num_steps,
        "base_seed": args.seed,
        "topk_definition": "Oracle metrics over the first k independently sampled candidates; candidates are not ranked by a model score.",
        "atom_correspondence": "mapped atom order; no permutation matching",
        "units": "angstrom",
        "metrics": topk_summary,
        "per_path_csv": str(per_path_csv.resolve()),
        "per_reference_csv": str(per_reference_csv.resolve()),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
