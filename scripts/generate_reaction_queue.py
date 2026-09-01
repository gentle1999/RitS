#!/usr/bin/env python3
"""Run a globally batched queue of reaction/seed TS sampling jobs."""

import argparse
import csv
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from torch.utils.data import Sampler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from megalodon.data.ts_batch_preprocessor import TsBatchPreProcessor
from megalodon.metrics.ts_evaluation_callback import convert_coords_to_np
from megalodon.models.module import Graph3DInterpolantModel
from sample_transition_state import coords_to_xyz_string, process_reaction_smarts


def load_reactions(
    path,
    kekulize,
    add_stereo,
    start_reaction=0,
    end_reaction=None,
):
    input_path = Path(path)
    if input_path.suffix.lower() == ".csv":
        with input_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "rxn_smiles" not in reader.fieldnames:
                raise ValueError(
                    f"CSV reaction input must contain an rxn_smiles column: {input_path}"
                )
            reactions = [
                (row.get("rxn_smiles") or "").strip()
                for row in reader
                if (row.get("rxn_smiles") or "").strip()
            ]
    else:
        reactions = [
            line.strip() for line in input_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    reaction_data = []
    for reaction_idx, reaction in enumerate(reactions):
        if reaction_idx < start_reaction:
            continue
        if end_reaction is not None and reaction_idx >= end_reaction:
            break
        reactant, product = reaction.split(">>")
        data = process_reaction_smarts(
            reactant,
            product,
            charge=0,
            kekulize=kekulize,
            add_stereo=add_stereo,
        )
        data.queue_reaction_idx = reaction_idx
        # Keep the exact source line for the XYZ comment and downstream audit.
        data.queue_reaction_smiles = reaction
        reaction_data.append(data)
    return reaction_data


def build_queue(
    reaction_data,
    n_samples,
    output_dir=None,
    skip_existing=False,
    queue_order="seed-major",
):
    queue = []
    if queue_order not in {"seed-major", "reaction-major"}:
        raise ValueError(f"unsupported queue_order={queue_order!r}")

    if queue_order == "seed-major":
        pairs = (
            (template, seed)
            for seed in range(n_samples)
            for template in reaction_data
        )
    else:
        # Keep all seeds of one reaction adjacent.  Atom-budget packing can
        # still combine several reaction groups into one GPU batch, preserving
        # the old rxn-level graph locality without giving up load balancing.
        pairs = (
            (template, seed)
            for template in reaction_data
            for seed in range(n_samples)
        )

    for template, seed in pairs:
        reaction_idx = int(template.queue_reaction_idx)
        if (
            skip_existing
            and (
                output_dir
                / f"seed_{seed}"
                / f"rxn_{reaction_idx:04d}.xyz"
            ).exists()
        ):
            continue
        item = deepcopy(template)
        # Preserve the original manifest index when a resumed run filters
        # out reactions whose seed files already exist.
        item.queue_reaction_idx = reaction_idx
        item.queue_seed = seed
        queue.append(item)
    return queue


def seeded_coordinate_prior(batch, device):
    node_counts = torch.bincount(batch.batch).tolist()
    seeds = batch.queue_seed.tolist()
    priors = []
    for seed, n_atoms in zip(seeds, node_counts):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        x0 = torch.randn((n_atoms, 3), generator=generator, device=device)
        priors.append(x0 - x0.mean(dim=0, keepdim=True))
    return torch.cat(priors, dim=0)


class AtomBudgetBatchSampler(Sampler):
    """Pack consecutive graphs until a total-atom budget is reached."""

    def __init__(self, dataset, max_atoms):
        self.dataset = dataset
        if max_atoms <= 0:
            raise ValueError("max_atoms must be positive")
        self.max_atoms = max_atoms

    def __iter__(self):
        batch = []
        atom_count = 0
        for index, data in enumerate(self.dataset):
            n_atoms = int(data.num_nodes)
            if n_atoms > self.max_atoms:
                raise ValueError(
                    f"graph {index} has {n_atoms} atoms, exceeding "
                    f"max_atoms_per_batch={self.max_atoms}"
                )
            if batch and atom_count + n_atoms > self.max_atoms:
                yield batch
                batch = []
                atom_count = 0
            batch.append(index)
            atom_count += n_atoms
        if batch:
            yield batch

    def __len__(self):
        return sum(1 for _ in self.__iter__())


def write_batch(output_dir, batch, coords_list):
    reaction_indices = batch.queue_reaction_idx.tolist()
    seeds = batch.queue_seed.tolist()
    reaction_smiles = batch.queue_reaction_smiles
    numbers = batch.numbers.cpu().numpy()
    graph_ids = batch.batch.cpu().numpy()
    for graph_idx, (reaction_idx, seed, coords) in enumerate(
        zip(reaction_indices, seeds, coords_list)
    ):
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        atom_numbers = numbers[graph_ids == graph_idx]
        xyz = coords_to_xyz_string(
            coords, atom_numbers, comment=f"rxn_smiles={reaction_smiles[graph_idx]}"
        )
        (seed_dir / f"rxn_{reaction_idx:04d}.xyz").write_text(xyz + "\n")


def batch_shape_stats(batch):
    """Return size statistics useful for diagnosing memory use."""
    atom_counts = torch.bincount(batch.batch)
    total_atoms = int(atom_counts.sum())
    sum_atom_squares = int(torch.sum(atom_counts * atom_counts))
    return {
        "graphs": int(atom_counts.numel()),
        "atoms": total_atoms,
        "max_atoms": int(atom_counts.max()),
        "sum_atom_squares": sum_atom_squares,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sample a flat global queue of reaction/seed pairs"
    )
    parser.add_argument("--reactions", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--start_reaction", type=int, default=0)
    parser.add_argument(
        "--end_reaction", type=int, default=None,
        help="Exclusive original manifest index; useful for disjoint GPU shards",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Maximum graphs per batch (use --max_atoms_per_batch for mixed-size queues)",
    )
    parser.add_argument(
        "--max_atoms_per_batch", type=int, default=None,
        help=(
            "Dynamic batch budget based on total flat atoms; "
            "90000 is a stable starting point on the tested 96 GiB GPU"
        ),
    )
    parser.add_argument(
        "--num_steps", type=int, default=16,
        help="ODE integration steps for inference (default: 16)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--no_write", action="store_true")
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip reactions with all seed_0..seed_(n_samples-1) XYZ files present",
    )
    parser.add_argument(
        "--queue_order",
        choices=("seed-major", "reaction-major"),
        default="seed-major",
        help=(
            "Queue ordering. reaction-major keeps a reaction's seeds adjacent "
            "for rxn-level graph locality; seed-major is the legacy global order."
        ),
    )
    parser.add_argument("--kekulize", action="store_true")
    parser.add_argument("--add_stereo", action="store_true")
    args = parser.parse_args()

    if not args.no_write and args.output_dir is None:
        parser.error("--output_dir is required unless --no_write is used")
    if args.start_reaction < 0:
        parser.error("--start_reaction must be non-negative")
    if args.end_reaction is not None and args.end_reaction < args.start_reaction:
        parser.error("--end_reaction must be greater than or equal to --start_reaction")
    if args.n_samples <= 0:
        parser.error("--n_samples must be positive")
    if args.batch_size is None and args.max_atoms_per_batch is None:
        parser.error("provide --batch_size or --max_atoms_per_batch")
    if args.batch_size is not None and args.max_atoms_per_batch is not None:
        parser.error("--batch_size and --max_atoms_per_batch are mutually exclusive")

    cfg = OmegaConf.load(args.config)
    batch_preprocessor = TsBatchPreProcessor(
        aug_rotations=cfg.data.get("aug_rotations", False),
        scale_coords=cfg.data.get("scale_coords", 1.0),
    )
    model = Graph3DInterpolantModel.load_from_checkpoint(
        args.ckpt,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=batch_preprocessor,
        strict=True,
    ).to(args.device).eval()

    reaction_data = load_reactions(
        args.reactions,
        args.kekulize,
        args.add_stereo,
        start_reaction=args.start_reaction,
        end_reaction=args.end_reaction,
    )
    output_dir = Path(args.output_dir) if args.output_dir else None
    if args.skip_existing:
        if output_dir is None:
            parser.error("--skip_existing requires --output_dir")
        before = len(reaction_data)
        reaction_data = [
            data for data in reaction_data
            if not all(
                (output_dir / f"seed_{seed}" / f"rxn_{int(data.queue_reaction_idx):04d}.xyz").exists()
                for seed in range(args.n_samples)
            )
        ]
        print(
            f"skip_existing: skipped {before - len(reaction_data)} complete reactions; "
            f"remaining {len(reaction_data)}",
            flush=True,
        )
    queue = build_queue(
        reaction_data,
        args.n_samples,
        output_dir=output_dir,
        skip_existing=args.skip_existing,
        queue_order=args.queue_order,
    )
    if args.skip_existing:
        print(
            f"skip_existing: remaining reaction/seed jobs={len(queue)}",
            flush=True,
        )
    print(f"queue_order={args.queue_order}", flush=True)
    if args.max_atoms_per_batch is not None:
        batch_sampler = AtomBudgetBatchSampler(queue, args.max_atoms_per_batch)
        loader = DataLoader(queue, batch_sampler=batch_sampler)
    else:
        loader = DataLoader(queue, batch_size=args.batch_size, shuffle=False)
    timesteps = args.num_steps or cfg.interpolant.timesteps
    total_start = time.perf_counter()
    completed = 0
    batch_stats = []
    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        batch = batch.to(args.device)
        initial_priors = {"ts_coord": seeded_coordinate_prior(batch, args.device)}
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
            torch.cuda.synchronize(args.device)
        shape_stats = batch_shape_stats(batch)
        print(
            f"batch={batch_idx} preparing graphs={shape_stats['graphs']} "
            f"flat_nodes={shape_stats['atoms']} "
            f"max_graph_atoms={shape_stats['max_atoms']} "
            f"sum_atom_squares={shape_stats['sum_atom_squares']}",
            flush=True,
        )
        start = time.perf_counter()
        with torch.no_grad():
            sample = model.sample(
                batch=batch,
                timesteps=timesteps,
                pre_format=True,
                initial_priors=initial_priors,
            )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
            peak_gib = torch.cuda.max_memory_allocated(args.device) / 1024 ** 3
        else:
            peak_gib = float("nan")
        elapsed = time.perf_counter() - start
        coords_list = convert_coords_to_np(sample)
        if output_dir is not None and not args.no_write:
            write_batch(output_dir, batch, coords_list)
        completed += len(coords_list)
        batch_stats.append((len(coords_list), elapsed, peak_gib))
        print(
            f"batch={batch_idx} graphs={len(coords_list)} "
            f"flat_nodes={batch.num_nodes} edges={sample['edge_index'].shape[1]} "
            f"seconds={elapsed:.3f} peak_allocated_gib={peak_gib:.3f}",
            flush=True,
        )

    total_elapsed = time.perf_counter() - total_start
    print(
        f"complete reactions={len(reaction_data)} queue={len(queue)} "
        f"sampled={completed} batches={len(batch_stats)} "
        f"sampling_wall_seconds={sum(x[1] for x in batch_stats):.3f} "
        f"loop_wall_seconds={total_elapsed:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
