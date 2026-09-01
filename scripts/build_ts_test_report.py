#!/usr/bin/env python3
"""Assemble the strict DPA test evaluation and final XYZ selection report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path


def parse_json_log(path: Path) -> dict:
    text = path.read_text()
    decoder = json.JSONDecoder()
    starts = [m.start() for m in re.finditer(r'\{\n  "checkpoint"', text)]
    for start in reversed(starts):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "checkpoint" in value:
            return value
    raise ValueError(f"no evaluation JSON found in {path}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number_stats(values: list[float]) -> dict:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"n": 0, "mean": None}
    n = len(finite)
    def percentile(q: float) -> float:
        index = (n - 1) * q
        left = int(index)
        right = min(left + 1, n - 1)
        fraction = index - left
        return finite[left] + fraction * (finite[right] - finite[left])
    mean = sum(finite) / n
    std = math.sqrt(sum((value - mean) ** 2 for value in finite) / n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": finite[0],
        "p25": percentile(0.25),
        "median": percentile(0.5),
        "p75": percentile(0.75),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "max": finite[-1],
        "fraction_le_0_5": sum(value <= 0.5 for value in finite) / n,
        "fraction_le_1_0": sum(value <= 1.0 for value in finite) / n,
        "fraction_le_2_0": sum(value <= 2.0 for value in finite) / n,
    }


def f(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dpa_ts_stereo_clean_split_test_final_report_seed48"),
    )
    args = parser.parse_args()

    project = Path(__file__).resolve().parent.parent
    output = args.output_dir
    selected_dir = output / "selected_xyz"
    rejected_dir = output / "rejected_candidates"
    selected_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    kabsch_log = project / "outputs/dpa_ts_stereo_clean_split_best_test_kabsch_16step_seed48.log"
    topk_log = project / "outputs/dpa_ts_stereo_clean_split_best_test_topk100_16step_seed48.log"
    xtb_root = project / "outputs/dpa_ts_stereo_clean_split_best_test_xtb_top100_seed48_allatom_thr0p5"
    stereo_root = xtb_root / "stereo_check_population_sign_only_no_n"
    baseline_path = project / "outputs/dpa_ts_stereo_clean_split_reference_internal_baseline/summary.json"

    kabsch = parse_json_log(kabsch_log)
    topk = parse_json_log(topk_log)
    xtb = json.loads((xtb_root / "summary.json").read_text())
    stereo = json.loads((stereo_root / "summary.json").read_text())
    baseline = json.loads(baseline_path.read_text())

    per_path = read_csv(xtb_root / "per_path.csv")
    accepted = read_csv(stereo_root / "accepted_top1_manifest.csv")
    rejected = read_csv(stereo_root / "rejected_paths.csv")
    candidates = read_csv(xtb_root / "per_candidate.csv")
    reactions = {
        row["reaction_id"]: row
        for row in read_csv(project / "data/new_full_df_exact_balanced_40ene_40diene.csv")
    }

    accepted_by_path = {row["path"]: row for row in accepted}
    rejected_by_path = {row["path"]: row for row in rejected}
    candidates_by_path: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        candidates_by_path.setdefault(row["path"], []).append(row)

    selected_stats = {
        "all_atom": number_stats(
            [float(row["nearest_reference_all_atom_rmsd"]) for row in accepted]
        ),
        "heavy_atom": number_stats(
            [float(row["nearest_reference_heavy_atom_rmsd"]) for row in accepted]
        ),
    }

    selection_rows = []
    for path_row in sorted(per_path, key=lambda row: int(row["global_path_index"])):
        path = path_row["path"]
        path_index = int(path_row["global_path_index"])
        reaction_id = path.removeprefix("reaction_id:")
        reaction = reactions.get(reaction_id, {})
        accepted_row = accepted_by_path.get(path)
        rejected_row = rejected_by_path.get(path)
        source_xyz = ""
        canonical_xyz = ""
        status = "accepted"
        if accepted_row:
            source_xyz = accepted_row["selected_xyz_path"]
            canonical_name = f"path_{path_index:04d}_{reaction_id}.xyz"
            canonical_path = selected_dir / canonical_name
            shutil.copy2(source_xyz, canonical_path)
            canonical_xyz = str(canonical_path.relative_to(output))
        else:
            status = "rejected_stereo"
            ranked = [
                row for row in candidates_by_path.get(path, [])
                if row.get("cluster_population_then_energy_rank") == "1"
            ]
            if ranked:
                candidate_index = int(ranked[0]["candidate_index"])
                source = (
                    xtb_root / "xyz" / "cluster_population_then_energy"
                    / f"path_{path_index:04d}_reaction_id_{reaction_id}__rank_01__candidate_{candidate_index:03d}.xyz"
                )
                if source.exists():
                    destination = rejected_dir / source.name
                    shutil.copy2(source, destination)
                    canonical_xyz = str(destination.relative_to(output))
                    source_xyz = str(source)
        selected_rmsd_all = accepted_row["nearest_reference_all_atom_rmsd"] if accepted_row else ""
        selected_rmsd_heavy = accepted_row["nearest_reference_heavy_atom_rmsd"] if accepted_row else ""
        selection_rows.append({
            "global_path_index": path_index,
            "path": path,
            "reaction_id": reaction_id,
            "ene_id": reaction.get("ene_id", ""),
            "diene_id": reaction.get("diene_id", ""),
            "prod_id": reaction.get("prod_id", ""),
            "rxn_smiles": reaction.get("rxn_smiles", ""),
            "status": status,
            "candidate_index": accepted_row.get("candidate_index", "") if accepted_row else "",
            "cluster": accepted_row.get("cluster", "") if accepted_row else "",
            "cluster_size": accepted_row.get("cluster_size", "") if accepted_row else "",
            "xtb_energy_hartree": accepted_row.get("xtb_energy_hartree", "") if accepted_row else "",
            "selected_nearest_reference_all_atom_rmsd": selected_rmsd_all,
            "selected_nearest_reference_heavy_atom_rmsd": selected_rmsd_heavy,
            "stereo_violations": rejected_row.get("violated_constraints", "") if rejected_row else "",
            "source_xyz": source_xyz,
            "xyz": canonical_xyz,
        })

    selection_csv = output / "path_selection.csv"
    with selection_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]))
        writer.writeheader()
        writer.writerows(selection_rows)

    final_selection = {
        "accepted": len(accepted),
        "total_paths": len(per_path),
        "coverage": len(accepted) / len(per_path),
        "rejected": [
            {
                "path": row["path"],
                "violated_constraints": row["violated_constraints"],
            }
            for row in rejected
        ],
        "accepted_rmsd": selected_stats,
        "pre_stereo_cluster_population_then_energy": xtb["strategies"]["cluster_population_then_energy"]["top1_best_all"],
        "pre_stereo_cluster_population_then_energy_heavy": xtb["strategies"]["cluster_population_then_energy"]["top1_best_heavy"],
    }

    metrics = {
        "model": {
            "checkpoint": kabsch["checkpoint"],
            "config": kabsch["config"],
            "dataset_root": kabsch["dataset_root"],
            "split": "test",
            "n_paths": kabsch["n_reactions"],
            "n_reference_conformers": kabsch["n_structures"],
            "num_steps": kabsch["num_steps"],
            "seed": kabsch["seed"],
            "atom_correspondence": kabsch["atom_correspondence"],
            "units": kabsch["units"],
        },
        "single_candidate_16step": {
            "all_atom": kabsch["all_atom_kabsch_rmsd"],
            "heavy_atom": kabsch["heavy_atom_kabsch_rmsd"],
            "path_mean_all_atom": kabsch["reaction_mean_all_atom_kabsch_rmsd"],
            "generated_to_nearest_reference_all_atom": kabsch["generated_to_nearest_reference_kabsch_rmsd"],
            "generated_to_nearest_reference_heavy_atom": kabsch["generated_to_nearest_reference_heavy_atom_kabsch_rmsd"],
            "reference_to_nearest_generated_all_atom": kabsch["reference_to_nearest_generated_kabsch_rmsd"],
            "reference_to_nearest_generated_heavy_atom": kabsch["reference_to_nearest_generated_heavy_atom_kabsch_rmsd"],
        },
        "topk_100_candidates_16step": {
            str(k): {
                "oracle_path_best_all_atom": topk["metrics"][f"top{k}"]["oracle_path_best_to_any_reference_all_atom_rmsd"],
                "oracle_path_best_heavy_atom": topk["metrics"][f"top{k}"]["oracle_path_best_to_any_reference_heavy_atom_rmsd"],
                "reference_coverage_all_atom": topk["metrics"][f"top{k}"]["reference_coverage_all_atom_rmsd_path_macro"],
                "reference_coverage_heavy_atom": topk["metrics"][f"top{k}"]["reference_coverage_heavy_atom_rmsd_path_macro"],
            }
            for k in topk["top_k"]
        },
        "reference_internal_baseline_clean_test": baseline,
        "cluster_xtb_stereo": {
            "cluster_method": xtb["cluster_method"],
            "cluster_threshold_angstrom": xtb["cluster_threshold_angstrom"],
            "cluster_counts": xtb["cluster_counts"],
            "xtb": xtb["xtb"],
            "stereo": stereo,
            "final_selection": final_selection,
        },
        "sources": {
            "kabsch_log": str(kabsch_log.resolve()),
            "topk_log": str(topk_log.resolve()),
            "xtb_summary": str((xtb_root / "summary.json").resolve()),
            "stereo_summary": str((stereo_root / "summary.json").resolve()),
            "path_selection_csv": str(selection_csv.resolve()),
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n")

    topk_metrics = metrics["topk_100_candidates_16step"]
    single = metrics["single_candidate_16step"]
    base = baseline["path_mean_nearest_other_all_atom_rmsd_macro"]
    base_heavy = baseline["path_mean_nearest_other_heavy_atom_rmsd_macro"]
    report = f"""# Strict DPA Test Report

This report evaluates the strictly split stereo-clean DPA test set with the
RitS fine-tuning checkpoint and exports the final per-path XYZ selection.

## Protocol

- Checkpoint: `{kabsch['checkpoint']}`
- Dataset/split: `{kabsch['dataset_root']}` / `test`
- Test size: {kabsch['n_reactions']} paths, {kabsch['n_structures']} reference conformers
- Inference: 16 steps, mapped atom order, no atom permutation, units Angstrom
- Candidate sweep: 100 independently sampled candidates per path
- Practical selection: complete-linkage all-atom Kabsch clustering at 0.5 Angstrom, GFN1-xTB single point on cluster medoids, population-then-energy ranking, then mapped stereo sign check

## Metrics

All RMSDs below are Kabsch RMSD in Angstrom. `oracle path-best` is an upper-bound
metric over the first k candidates; `reference coverage` is the path-macro mean
of each reference's nearest generated candidate.

| Evaluation | All atom mean | Heavy atom mean |
|---|---:|---:|
| Single candidate, 16 steps | {f(single['all_atom']['mean'])} | {f(single['heavy_atom']['mean'])} |
| Single generated -> nearest reference | {f(single['generated_to_nearest_reference_all_atom']['mean'])} | {f(single['generated_to_nearest_reference_heavy_atom']['mean'])} |
| Reference -> nearest single generated | {f(single['reference_to_nearest_generated_all_atom']['mean'])} | {f(single['reference_to_nearest_generated_heavy_atom']['mean'])} |
| Internal reference baseline, nearest other conformer (path macro) | {f(base['mean'])} | {f(base_heavy['mean'])} |
| Final selection before stereo rejection, 298 paths | {f(final_selection['pre_stereo_cluster_population_then_energy']['mean'])} | {f(final_selection['pre_stereo_cluster_population_then_energy_heavy']['mean'])} |
| Final accepted selection, 296 paths | {f(selected_stats['all_atom']['mean'])} | {f(selected_stats['heavy_atom']['mean'])} |

| Candidates | Oracle path-best all atom | Oracle path-best heavy atom | Reference coverage all atom | Reference coverage heavy atom |
|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| Top{k} | {f(topk_metrics[str(k)]['oracle_path_best_all_atom']['mean'])} | {f(topk_metrics[str(k)]['oracle_path_best_heavy_atom']['mean'])} | {f(topk_metrics[str(k)]['reference_coverage_all_atom']['mean'])} | {f(topk_metrics[str(k)]['reference_coverage_heavy_atom']['mean'])} |"
        for k in topk["top_k"]
    ) + f"""

## Screening and Stereo

- Clusters: mean {xtb['cluster_counts']['mean']:.2f} per path at the 0.5 Angstrom threshold.
- xTB: {xtb['xtb']['successes']}/{xtb['xtb']['total']} medoid single points succeeded; mean wall time {xtb['xtb']['wall_seconds_per_candidate']['mean']:.4f} s per medoid.
- Stereo constraints: {stereo['constraints']['n_constraint_definitions']} checked definitions, with nitrogen-centered tetrahedral and E/Z constraints ignored as requested.
- Final accepted XYZ: {len(accepted)}/{len(per_path)} paths ({len(accepted) / len(per_path):.2%}).
- Rejected paths: `rxn-00000073` (reactant E/Z sign reversals) and `rxn-00002520` (product tetrahedral sign reversal).

The two rejected paths intentionally have no final accepted XYZ. Their rejected
top-ranked candidates are copied under `rejected_candidates/` for inspection.

## Files

- `selected_xyz/`: one canonical XYZ file per accepted path, named `path_XXXX_rxn-YYYYYYYY.xyz`.
- `path_selection.csv`: all 298 paths, reaction IDs, DPA IDs, mapped reaction SMILES, selected candidate, RMSDs, stereo status, and XYZ paths.
- `metrics.json`: machine-readable copy of the report metrics and source summaries.

The source per-candidate and stereo audit tables remain in:
`{xtb_root.resolve()}`.
"""
    (output / "report.md").write_text(report)
    print(json.dumps({
        "report": str((output / "report.md").resolve()),
        "metrics": str((output / "metrics.json").resolve()),
        "path_selection": str(selection_csv.resolve()),
        "selected_xyz_dir": str(selected_dir.resolve()),
        "selected_paths": len(accepted),
        "total_paths": len(per_path),
        "rejected_paths": len(rejected),
    }, indent=2))


if __name__ == "__main__":
    main()
