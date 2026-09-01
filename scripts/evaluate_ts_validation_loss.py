#!/usr/bin/env python3
"""Evaluate TS flow-matching validation loss with reproducible random draws."""

import argparse
import json
import random
import sys
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
from megalodon.models.module import Graph3DInterpolantModel


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_once(model, dataset, batch_preprocessor, batch_size, seed, device):
    seed_everything(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    weighted_loss = 0.0
    n_graphs_total = 0
    batch_losses = []

    for batch in tqdm(loader, desc=f"validation loss seed={seed}"):
        n_graphs = int(batch.num_graphs)
        batch = batch.to(device)
        batch = batch_preprocessor(batch)
        with torch.inference_mode():
            time = model.sample_time(batch)
            out, batch, time = model(batch, time)
            weights = model.interpolants[model.global_variable].loss_weight_t(time)
            total_loss = torch.zeros((), device=device)
            for key, loss_fn in model.loss_functions.items():
                target_key = f"{key}_target" if f"{key}_target" in batch else key
                if not loss_fn.continuous or "edge" in key:
                    raise ValueError(
                        "This evaluator currently supports continuous non-edge TS losses only"
                    )
                sub_loss, _ = loss_fn(
                    batch.batch,
                    out[f"{key}_hat"],
                    batch[target_key],
                    batch_weight=weights,
                    level=1.0,
                )
                total_loss += sub_loss

        loss_value = float(total_loss.item())
        batch_losses.append(loss_value)
        weighted_loss += loss_value * n_graphs
        n_graphs_total += n_graphs

    return {
        "seed": seed,
        "loss": weighted_loss / n_graphs_total,
        "n_batches": len(batch_losses),
        "batch_loss_mean": float(np.mean(batch_losses)),
        "batch_loss_std": float(np.std(batch_losses)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--processed_folder", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[48])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset_root = args.dataset_root or Path(cfg.data.dataset_root)
    processed_folder = args.processed_folder or cfg.data.processed_folder
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
        root=str(dataset_root), processed_folder=processed_folder, split=args.split
    )
    if args.max_samples is not None:
        dataset = dataset[: min(args.max_samples, len(dataset))]

    results = [
        evaluate_once(
            model, dataset, batch_preprocessor, args.batch_size, seed, args.device
        )
        for seed in args.seeds
    ]
    losses = [result["loss"] for result in results]
    summary = {
        "checkpoint": str(args.ckpt.resolve()),
        "config": str(args.config.resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
        "processed_folder": str(processed_folder),
        "split": args.split,
        "n_structures": len(dataset),
        "batch_size": args.batch_size,
        "loss_clamp": 1.0,
        "aggregation": "Lightning-compatible graph-count-weighted mean of batch losses",
        "seed_results": results,
        "loss_mean": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
