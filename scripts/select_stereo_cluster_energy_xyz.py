#!/usr/bin/env python3
"""Select stereo-valid minimum-energy representatives from TS candidate clusters."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_ts_stereo_constraints import (
    evaluate_coords,
    extract_reaction_constraints,
    load_reactions,
    reaction_for_path,
)
from evaluate_ts_xtb_ranking import (
    cluster_candidates,
    pairwise_kabsch_rmsd,
    write_xyz,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None}
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def fmt(value: float | None, digits: int = 4) -> str:
    return "无" if value is None else f"{value:.{digits}f}"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def assemble_metrics(
    single_path: Path, candidate_path: Path, baseline_path: Path
) -> dict:
    """Normalize standalone evaluation summaries to the report schema."""
    single = json.loads(single_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    generation = candidate["strategies"]["generation_order"]
    topk = {}
    for k in (1, 5, 10, 50, 100):
        prefix = f"top{k}"
        topk[str(k)] = {
            "oracle_path_best_all_atom": generation[f"{prefix}_best_all"],
            "oracle_path_best_heavy_atom": generation[f"{prefix}_best_heavy"],
            "reference_coverage_all_atom": generation[f"{prefix}_coverage_all"],
            "reference_coverage_heavy_atom": generation[
                f"{prefix}_coverage_heavy"
            ],
        }
    return {
        "model": {
            "checkpoint": single["checkpoint"],
            "config": single["config"],
            "dataset_root": single["dataset_root"],
            "split": single["split"],
            "n_paths": single["n_reactions"],
            "n_reference_conformers": single["n_structures"],
            "num_steps": single["num_steps"],
            "seed": single["seed"],
            "atom_correspondence": single["atom_correspondence"],
            "units": single["units"],
        },
        "single_candidate_16step": {
            "all_atom": single["all_atom_kabsch_rmsd"],
            "heavy_atom": single["heavy_atom_kabsch_rmsd"],
            "path_mean_all_atom": single[
                "reaction_mean_all_atom_kabsch_rmsd"
            ],
            "generated_to_nearest_reference_all_atom": single[
                "generated_to_nearest_reference_kabsch_rmsd"
            ],
            "generated_to_nearest_reference_heavy_atom": single[
                "generated_to_nearest_reference_heavy_atom_kabsch_rmsd"
            ],
            "reference_to_nearest_generated_all_atom": single[
                "reference_to_nearest_generated_kabsch_rmsd"
            ],
            "reference_to_nearest_generated_heavy_atom": single[
                "reference_to_nearest_generated_heavy_atom_kabsch_rmsd"
            ],
        },
        "topk_100_candidates_16step": topk,
        "reference_internal_baseline_clean_test": baseline,
        "sources": {
            "single_candidate_summary": str(single_path.resolve()),
            "candidate_xtb_summary": str(candidate_path.resolve()),
            "reference_internal_baseline": str(baseline_path.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--per-candidate", type=Path, required=True)
    parser.add_argument("--cluster-summary", type=Path, required=True)
    parser.add_argument("--reaction-csv", type=Path, required=True)
    parser.add_argument("--threshold-sweep", type=Path, required=True)
    parser.add_argument("--test-metrics", type=Path)
    parser.add_argument("--single-metrics", type=Path)
    parser.add_argument("--candidate-metrics", type=Path)
    parser.add_argument("--reference-baseline", type=Path)
    parser.add_argument("--model-label", default="RitS 模型")
    parser.add_argument("--cluster-atoms", choices=("heavy", "all"), default="heavy")
    parser.add_argument("--cluster-threshold", type=float, default=1.0)
    parser.add_argument("--energy-threshold", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    xyz_root = args.output_dir / "final_xyz"
    xyz_root.mkdir(exist_ok=True)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    is_zero_shot = Path(cache["checkpoint"]).resolve() == (
        PROJECT_ROOT / "data" / "rits.ckpt"
    ).resolve()
    evaluation_mode = "zero-shot" if is_zero_shot else "fine-tuned"
    evaluation_description = (
        "zero-shot；使用发布的 RitS 权重，未在 DPA TS 数据上做任何梯度更新"
        if is_zero_shot
        else "DPA TS 微调 checkpoint"
    )
    records = {record["path"]: record for record in cache["records"]}
    candidate_rows_by_path: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.per_candidate):
        candidate_rows_by_path[row["path"]].append(row)
    by_reaction_id, by_dpa_key = load_reactions(args.reaction_csv)

    audit_rows = []
    cluster_rows = []
    final_rows = []
    path_rows = []
    rejected_paths = []
    mapping_failures = []

    for record in sorted(records.values(), key=lambda value: value["global_path_index"]):
        path = record["path"]
        reaction = reaction_for_path(path, by_reaction_id, by_dpa_key)
        constraints, skipped, expected_numbers = extract_reaction_constraints(
            reaction["rxn_smiles"]
        )
        numbers = record["numbers"].detach().cpu().numpy().astype(np.int64)
        mapping_ok = bool(np.array_equal(numbers, expected_numbers))
        if not mapping_ok:
            mapping_failures.append(path)
        rows = sorted(
            candidate_rows_by_path[path], key=lambda row: int(row["candidate_index"])
        )
        if len(rows) != len(record["candidates"]):
            raise ValueError(f"candidate count mismatch for {path}")

        evaluated = []
        for row in rows:
            candidate_index = int(row["candidate_index"])
            coords = record["candidates"][candidate_index].detach().cpu().numpy()
            passed, details = evaluate_coords(coords, constraints)
            passed = bool(mapping_ok and passed)
            violations = [detail["constraint"] for detail in details if not detail["pass"]]
            energy = float(row["energy_hartree"])
            energy_valid = bool(int(row["xtb_success"]) and math.isfinite(energy))
            evaluated_row = {
                "path": path,
                "global_path_index": record["global_path_index"],
                "reaction_id": reaction["reaction_id"],
                "candidate_index": candidate_index,
                "stereo_filtered_cluster": "",
                "xtb_success": int(energy_valid),
                "xtb_energy_hartree": f"{energy:.12f}" if energy_valid else "",
                "nearest_reference_all_atom_rmsd": row["nearest_reference_all_atom_rmsd"],
                "nearest_reference_heavy_atom_rmsd": row["nearest_reference_heavy_atom_rmsd"],
                "mapping_order_valid": int(mapping_ok),
                "n_constraints": len(constraints),
                "n_skipped_constraints": len(skipped),
                "pass_all_stereo": int(passed),
                "violations": ";".join(violations),
            }
            evaluated.append(evaluated_row)
            audit_rows.append(evaluated_row)

        # Stereo validity defines the clustering population. xTB success is
        # deliberately not part of this filter; it is used only when choosing
        # an energy-ranked representative from each resulting cluster.
        stereo_valid = [row for row in evaluated if int(row["pass_all_stereo"])]
        grouped_clusters: dict[int, list[dict]] = defaultdict(list)
        if stereo_valid:
            valid_indices = [int(row["candidate_index"]) for row in stereo_valid]
            valid_coords = record["candidates"][valid_indices].detach().cpu()
            atom_mask = (
                torch.as_tensor(numbers > 1)
                if args.cluster_atoms == "heavy"
                else torch.ones(len(numbers), dtype=torch.bool)
            )
            distance_matrix = pairwise_kabsch_rmsd(valid_coords, atom_mask)
            labels = cluster_candidates(distance_matrix, args.cluster_threshold)
            for row, label in zip(stereo_valid, labels, strict=True):
                cluster = int(label)
                row["stereo_filtered_cluster"] = cluster
                grouped_clusters[cluster].append(row)

        representatives = []
        for cluster, members in sorted(grouped_clusters.items()):
            eligible = [row for row in members if int(row["xtb_success"])]
            representative = (
                min(eligible, key=lambda row: float(row["xtb_energy_hartree"]))
                if eligible
                else None
            )
            cluster_row = {
                "path": path,
                "global_path_index": record["global_path_index"],
                "reaction_id": reaction["reaction_id"],
                "cluster": cluster,
                "cluster_size": len(members),
                "n_stereo_valid_members": len(members),
                "n_xtb_successful_members": len(eligible),
                "representative_candidate_index": (
                    representative["candidate_index"] if representative else ""
                ),
                "representative_energy_hartree": (
                    representative["xtb_energy_hartree"] if representative else ""
                ),
                "representative_nearest_reference_all_atom_rmsd": (
                    representative["nearest_reference_all_atom_rmsd"] if representative else ""
                ),
                "representative_nearest_reference_heavy_atom_rmsd": (
                    representative["nearest_reference_heavy_atom_rmsd"] if representative else ""
                ),
                "has_valid_representative": int(representative is not None),
                "path_energy_min_hartree": "",
                "energy_delta_hartree": "",
                "within_energy_threshold": 0,
                "xyz_path": "",
            }
            cluster_rows.append(cluster_row)
            if representative:
                representatives.append((representative, cluster_row))

        if not representatives:
            rejected_paths.append(
                {
                    "path": path,
                    "global_path_index": record["global_path_index"],
                    "reaction_id": reaction["reaction_id"],
                    "reason": "no stereo-valid candidate with successful xTB energy",
                }
            )
            path_rows.append(
                {
                    "path": path,
                    "global_path_index": record["global_path_index"],
                    "reaction_id": reaction["reaction_id"],
                    "ene_id": reaction["ene_id"],
                    "diene_id": reaction["diene_id"],
                    "prod_id": reaction["prod_id"],
                    "n_clusters": len(grouped_clusters),
                    "n_clusters_with_valid_representative": 0,
                    "n_final_xyz": 0,
                    "path_energy_min_hartree": "",
                    "path_best_all_atom_rmsd": "",
                    "path_best_heavy_atom_rmsd": "",
                    "rxn_smiles": reaction["rxn_smiles"],
                }
            )
            continue

        path_minimum = min(float(item[0]["xtb_energy_hartree"]) for item in representatives)
        ranked = sorted(representatives, key=lambda item: float(item[0]["xtb_energy_hartree"]))
        selected = []
        for energy_rank, (representative, cluster_row) in enumerate(ranked, start=1):
            energy = float(representative["xtb_energy_hartree"])
            delta = energy - path_minimum
            keep = delta <= args.energy_threshold + 1e-12
            cluster_row["path_energy_min_hartree"] = f"{path_minimum:.12f}"
            cluster_row["energy_delta_hartree"] = f"{delta:.12f}"
            cluster_row["within_energy_threshold"] = int(keep)
            if not keep:
                continue
            candidate_index = int(representative["candidate_index"])
            cluster = int(representative["stereo_filtered_cluster"])
            path_dir = xyz_root / (
                f"path_{record['global_path_index']:04d}_{safe_name(path)}"
            )
            path_dir.mkdir(exist_ok=True)
            xyz_path = path_dir / (
                f"cluster_{cluster:03d}__energy_rank_{energy_rank:02d}"
                f"__candidate_{candidate_index:03d}.xyz"
            )
            comment = (
                f"path={path} cluster={cluster} cluster_size={cluster_row['cluster_size']} "
                f"candidate_index={candidate_index} energy_rank={energy_rank} "
                f"xtb_energy_hartree={energy:.12f} energy_delta_hartree={delta:.12f} "
                f"energy_threshold_hartree={args.energy_threshold:.12f} stereo_valid=1"
            )
            write_xyz(
                xyz_path,
                record["numbers"].tolist(),
                record["candidates"][candidate_index].tolist(),
                record["charge"],
                comment=comment,
            )
            cluster_row["xyz_path"] = str(xyz_path.resolve())
            final_row = {
                "path": path,
                "global_path_index": record["global_path_index"],
                "reaction_id": reaction["reaction_id"],
                "ene_id": reaction["ene_id"],
                "diene_id": reaction["diene_id"],
                "prod_id": reaction["prod_id"],
                "cluster": cluster,
                "cluster_size": cluster_row["cluster_size"],
                "candidate_index": candidate_index,
                "energy_rank": energy_rank,
                "xtb_energy_hartree": f"{energy:.12f}",
                "energy_delta_hartree": f"{delta:.12f}",
                "energy_threshold_hartree": f"{args.energy_threshold:.12f}",
                "nearest_reference_all_atom_rmsd": representative["nearest_reference_all_atom_rmsd"],
                "nearest_reference_heavy_atom_rmsd": representative["nearest_reference_heavy_atom_rmsd"],
                "n_constraints": representative["n_constraints"],
                "n_skipped_constraints": representative["n_skipped_constraints"],
                "pass_all_stereo": 1,
                "xyz_path": str(xyz_path.resolve()),
                "rxn_smiles": reaction["rxn_smiles"],
            }
            final_rows.append(final_row)
            selected.append(final_row)

        path_rows.append(
            {
                "path": path,
                "global_path_index": record["global_path_index"],
                "reaction_id": reaction["reaction_id"],
                "ene_id": reaction["ene_id"],
                "diene_id": reaction["diene_id"],
                "prod_id": reaction["prod_id"],
                "n_clusters": len(grouped_clusters),
                "n_clusters_with_valid_representative": len(representatives),
                "n_final_xyz": len(selected),
                "path_energy_min_hartree": f"{path_minimum:.12f}",
                "path_best_all_atom_rmsd": min(
                    float(row["nearest_reference_all_atom_rmsd"]) for row in selected
                ),
                "path_best_heavy_atom_rmsd": min(
                    float(row["nearest_reference_heavy_atom_rmsd"]) for row in selected
                ),
                "rxn_smiles": reaction["rxn_smiles"],
            }
        )

    write_csv(args.output_dir / "candidate_stereo_energy.csv", audit_rows)
    write_csv(args.output_dir / "cluster_representatives.csv", cluster_rows)
    write_csv(args.output_dir / "final_xyz_manifest.csv", final_rows)
    write_csv(args.output_dir / "path_summary.csv", path_rows)
    write_csv(args.output_dir / "rejected_paths.csv", rejected_paths)

    accepted_path_rows = [row for row in path_rows if int(row["n_final_xyz"]) > 0]
    cluster_summary = json.loads(args.cluster_summary.read_text())
    summary = {
        "checkpoint": cache["checkpoint"],
        "evaluation_mode": evaluation_mode,
        "dataset_split": cache["split"],
        "n_paths": len(path_rows),
        "n_candidates": len(audit_rows),
        "n_xtb_success": sum(int(row["xtb_success"]) for row in audit_rows),
        "n_stereo_valid_candidates": sum(int(row["pass_all_stereo"]) for row in audit_rows),
        "mapping_failure_paths": mapping_failures,
        "cluster_input": "stereo-valid candidates only",
        "cluster_method": (
            f"complete-linkage {args.cluster_atoms}-atom Kabsch RMSD"
        ),
        "cluster_atoms": args.cluster_atoms,
        "cluster_threshold_angstrom": args.cluster_threshold,
        "n_clusters": len(cluster_rows),
        "n_clusters_with_stereo_valid_representative": sum(
            int(row["has_valid_representative"]) for row in cluster_rows
        ),
        "energy_method": cluster_summary["xtb"]["method"],
        "energy_threshold_hartree": args.energy_threshold,
        "energy_threshold_kcal_per_mol": args.energy_threshold * 627.509474,
        "n_final_xyz": len(final_rows),
        "n_paths_with_final_xyz": len(accepted_path_rows),
        "n_rejected_paths": len(rejected_paths),
        "rejected_paths": rejected_paths,
        "final_xyz_per_accepted_path": metric(
            int(row["n_final_xyz"]) for row in accepted_path_rows
        ),
        "final_candidate_nearest_reference_all_atom_rmsd": metric(
            float(row["nearest_reference_all_atom_rmsd"]) for row in final_rows
        ),
        "final_candidate_nearest_reference_heavy_atom_rmsd": metric(
            float(row["nearest_reference_heavy_atom_rmsd"]) for row in final_rows
        ),
        "final_path_best_all_atom_rmsd": metric(
            float(row["path_best_all_atom_rmsd"]) for row in accepted_path_rows
        ),
        "final_path_best_heavy_atom_rmsd": metric(
            float(row["path_best_heavy_atom_rmsd"]) for row in accepted_path_rows
        ),
        "outputs": {
            "final_xyz_dir": str(xyz_root.resolve()),
            "final_xyz_manifest": str((args.output_dir / "final_xyz_manifest.csv").resolve()),
            "path_summary": str((args.output_dir / "path_summary.csv").resolve()),
            "cluster_representatives": str((args.output_dir / "cluster_representatives.csv").resolve()),
            "candidate_audit": str((args.output_dir / "candidate_stereo_energy.csv").resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    )

    if args.test_metrics:
        metrics = json.loads(args.test_metrics.read_text())
    elif args.single_metrics and args.candidate_metrics and args.reference_baseline:
        metrics = assemble_metrics(
            args.single_metrics, args.candidate_metrics, args.reference_baseline
        )
    else:
        parser.error(
            "provide --test-metrics, or all of --single-metrics, "
            "--candidate-metrics and --reference-baseline"
        )
    (args.output_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n"
    )
    sweep = json.loads(args.threshold_sweep.read_text())
    single = metrics["single_candidate_16step"]
    topk = metrics["topk_100_candidates_16step"]
    baseline = metrics["reference_internal_baseline_clean_test"]
    chosen = sweep["by_atoms"][args.cluster_atoms][str(args.cluster_threshold)]
    all_one = sweep["by_atoms"]["all"]["1.0"]
    xyz_count_histogram = Counter(int(row["n_final_xyz"]) for row in path_rows)
    xyz_histogram_rows = "\n".join(
        f"| {count} | {xyz_count_histogram[count]} | "
        f"{xyz_count_histogram[count] / len(path_rows):.1%} |"
        for count in sorted(xyz_count_histogram)
    )
    rejected_path_text = (
        "、".join(f"`{row['path']}`" for row in rejected_paths)
        if rejected_paths
        else "无"
    )
    baseline_all = baseline["path_mean_nearest_other_all_atom_rmsd_macro"]["mean"]
    baseline_heavy = baseline[
        "path_mean_nearest_other_heavy_atom_rmsd_macro"
    ]["mean"]
    final_all = summary["final_path_best_all_atom_rmsd"]["mean"]
    final_heavy = summary["final_path_best_heavy_atom_rmsd"]["mean"]
    if final_all <= baseline_all and final_heavy <= baseline_heavy:
        baseline_interpretation = (
            "两项实际 path-best 均不高于内部 path-macro 基准，说明最终集合通常"
            "至少包含一个处在已知参考内部变化尺度内的候选"
        )
    else:
        baseline_interpretation = (
            "实际 path-best 比内部 path-macro 基准高出全原子 "
            f"{final_all - baseline_all:.4f} Å、重原子 "
            f"{final_heavy - baseline_heavy:.4f} Å；按这一统计口径，当前模型和筛选"
            "流程尚未达到参考构象内部基准"
        )
    report = f"""# {args.model_label}测试集多构象报告

## 摘要

本报告评估严格按 path 划分的 stereo-clean DPA 测试集。实际可用的结果不是
Top100 oracle，而是经过“立体检查 → 聚类 → 类内 xTB 最低能量 → path 内能窗”
得到的最终多 XYZ 集合。

- 最终集合覆盖 {summary['n_paths_with_final_xyz']}/{summary['n_paths']} 条 path，共 {summary['n_final_xyz']} 个 XYZ。
- 每条 path 至少一个候选时的实际 `path-best`：全原子 {fmt(summary['final_path_best_all_atom_rmsd']['mean'])} Å，重原子 {fmt(summary['final_path_best_heavy_atom_rmsd']['mean'])} Å。
- 100 候选 oracle 上限：全原子 {fmt(topk['100']['oracle_path_best_all_atom']['mean'])} Å，重原子 {fmt(topk['100']['oracle_path_best_heavy_atom']['mean'])} Å。该指标使用真实参考结构挑选候选，不能用于实际筛选。
- 测试参考构象的内部最近邻 path-macro 基准：全原子 {fmt(baseline['path_mean_nearest_other_all_atom_rmsd_macro']['mean'])} Å，重原子 {fmt(baseline['path_mean_nearest_other_heavy_atom_rmsd_macro']['mean'])} Å。

## 数据与模型

- 模型：`{cache['checkpoint']}`
- 评估模式：{evaluation_description}。
- 数据：`{metrics['model']['dataset_root']}` / `{cache['split']}`。
- 测试集：{len(path_rows)} 条 path、{baseline['n_conformers']} 个参考构象；其中 {baseline['n_multiconformer_paths']} 条多构象 path、{baseline['n_singleton_paths']} 条单构象 path。
- 推理：每条 path 独立生成 100 个候选，共 {len(audit_rows)} 个；每个候选 16 个推理步。
- 坐标单位：Å；原子对应关系固定为 atom mapping 顺序，不允许原子置换匹配。

## RMSD 计算口径

Kabsch RMSD 先移除两套坐标的整体平移，并寻找使均方距离最小的刚体旋转；
因此它衡量内部几何差异，不受分子在 XYZ 中朝向和位置的影响。对齐不允许镜像反射，
也不重新排列同种元素；第 `i` 个坐标始终和 atom map 为 `i+1` 的原子对应。

- **全原子 RMSD**：包含氢原子，对 X-H 朝向、甲基转动和质子位置更敏感。
- **重原子 RMSD**：只使用原子序数大于 1 的原子，更直接反映反应骨架和取代基构象。
- **立体合法性不是 RMSD 的一部分**：镜像或 E/Z 反号可能仍有不大的 RMSD，因此本流程在聚类前另做带符号立体约束检查。
- 所有表中的 `mean` 都是先按该行规定的层级取最小值，再求平均；不同层级的均值不能直接当成同一个分数比较。

## 最终筛选流程

1. 先对全部 {len(audit_rows)} 个候选逐一做立体检查，只剔除明确反号；接近零、零或无法定义的 TS 手性量不拒绝。N 中心四面体和 N 端点 E/Z 不约束，N 作为其他中心的取代基仍参与约束。
2. 只对通过立体检查的候选重新计算{('重原子' if args.cluster_atoms == 'heavy' else '全原子')} Kabsch RMSD，并做 complete-linkage 聚类，阈值为 {args.cluster_threshold:.1f} Å。旧的全候选 cluster 标签完全不参与最终筛选。
3. 读取已有 GFN1-xTB 单点能；全部候选中成功 {summary['n_xtb_success']}/{len(audit_rows)}。xTB 是否成功不改变立体过滤结果，只决定该成员能否参加类内能量排序。
4. 每个立体过滤后得到的 cluster 选择 xTB 能量最低成员，确保每个保留 cluster 只输出一个代表。
5. 以每条 path 的最低 cluster 代表能量为零点，只保留 ΔE ≤ {args.energy_threshold:.3f} Eh（{args.energy_threshold * 627.509474:.3f} kcal/mol）的代表。

## 筛选漏斗

| 阶段 | 数量 | 说明 |
|---|---:|---|
| 原始生成候选 | {len(audit_rows)} | {len(path_rows)} paths × 100 |
| 立体合法候选 | {summary['n_stereo_valid_candidates']} | 删除 {len(audit_rows) - summary['n_stereo_valid_candidates']} 个明确反号候选 |
| 立体过滤后 clusters | {summary['n_clusters']} | 重原子 {args.cluster_threshold:.1f} Å complete-linkage |
| 有 xTB 能量代表的 clusters | {summary['n_clusters_with_stereo_valid_representative']} | 每类取最低能成员 |
| 能窗内最终 XYZ | {summary['n_final_xyz']} | ΔE ≤ {args.energy_threshold:.3f} Eh |
| 有最终 XYZ 的 paths | {summary['n_paths_with_final_xyz']} | 共 {summary['n_paths']} 条测试 path |

## 聚类阈值选择

`0.5 Å` 全原子聚类平均产生 {sweep['by_atoms']['all']['0.5']['mean']:.2f} 类/path，主要放大了氢原子的小幅变化。
阈值扫描后选择{('重原子' if args.cluster_atoms == 'heavy' else '全原子')} `{args.cluster_threshold:.1f} Å`。
下表是对原始 100 候选所做的阈值扫描，用于选择参数；最终的 {summary['n_clusters']} 个 cluster
是在立体过滤后重新计算的，因此最终平均类数为 {summary['n_clusters'] / len(path_rows):.2f}/path，
不应要求与扫描表完全一致。

| 聚类定义 | 平均类数/path | 中位数 | P90 | 最大值 |
|---|---:|---:|---:|---:|
| 全原子 1.0 Å | {all_one['mean']:.2f} | {all_one['median']:.1f} | {all_one['p90']:.1f} | {all_one['max']} |
| 重原子 1.0 Å（参数选择） | {chosen['mean']:.2f} | {chosen['median']:.1f} | {chosen['p90']:.1f} | {chosen['max']} |

## 最终多构象集合

- 立体检查通过：{summary['n_stereo_valid_candidates']}/{len(audit_rows)} 个候选；明确反号并被剔除：{len(audit_rows) - summary['n_stereo_valid_candidates']} 个。
- 能窗过滤后最终 XYZ：{summary['n_final_xyz']} 个，覆盖 {summary['n_paths_with_final_xyz']}/{summary['n_paths']} 条 path。
- 每条成功 path 的 XYZ 数：平均 {summary['final_xyz_per_accepted_path']['mean']:.2f}，中位数 {summary['final_xyz_per_accepted_path']['median']:.1f}，P90 为 {summary['final_xyz_per_accepted_path']['p90']:.1f}，范围 {summary['final_xyz_per_accepted_path']['min']:.0f}–{summary['final_xyz_per_accepted_path']['max']:.0f}。
- 没有最终合法候选的 path：{rejected_path_text}。

| 每条 path 的最终 XYZ 数 | path 数 | 占全部测试 path |
|---:|---:|---:|
{xyz_histogram_rows}

## 测试集结构误差：指标定义

下表中的“比较对象数”也是求均值时的样本数。四行指标回答的是四个不同问题：

| 指标 | 比较和取最小值的方式 | 比较对象数 | 回答的问题 |
|---|---|---:|---|
| 单候选、16 步 | 每个生成结构只和其数据行对应的参考构象比较 | {baseline['n_conformers']} 个测试结构 | 固定标签下，单次生成与指定参考相差多远？ |
| 单候选到同 path 最近参考 | 每个生成结构对同 path 所有参考取最小 RMSD | {baseline['n_conformers']} 个生成结构 | 承认同一 TS 有多构象后，单次生成是否像其中任意一个？ |
| 最终全部 XYZ 到最近参考 | 每个最终 XYZ 对同 path 所有参考取最小 RMSD | {summary['n_final_xyz']} 个最终 XYZ | 实际输出的每个结构平均有多像参考集合？ |
| 最终集合 path-best | 每条 path 在“最终 XYZ × 全部参考”矩阵中取全局最小值 | {len(path_rows)} 条 path | 实际输出集合中，每条 path 是否至少有一个很好的候选？ |

### 均值结果

| 指标 | 全原子均值 | 重原子均值 |
|---|---:|---:|
| 单候选、16 步 | {fmt(single['all_atom']['mean'])} | {fmt(single['heavy_atom']['mean'])} |
| 单候选到同 path 最近参考 | {fmt(single['generated_to_nearest_reference_all_atom']['mean'])} | {fmt(single['generated_to_nearest_reference_heavy_atom']['mean'])} |
| 最终保留的全部 XYZ 到最近参考 | {fmt(summary['final_candidate_nearest_reference_all_atom_rmsd']['mean'])} | {fmt(summary['final_candidate_nearest_reference_heavy_atom_rmsd']['mean'])} |
| 最终集合 path-best | {fmt(summary['final_path_best_all_atom_rmsd']['mean'])} | {fmt(summary['final_path_best_heavy_atom_rmsd']['mean'])} |

### 分布而不只是均值

| 指标 | 全原子 median / P90 / max | 重原子 median / P90 / max |
|---|---:|---:|
| 单候选、16 步 | {fmt(single['all_atom']['median'])} / {fmt(single['all_atom']['p90'])} / {fmt(single['all_atom']['max'])} | {fmt(single['heavy_atom']['median'])} / {fmt(single['heavy_atom']['p90'])} / {fmt(single['heavy_atom']['max'])} |
| 单候选到同 path 最近参考 | {fmt(single['generated_to_nearest_reference_all_atom']['median'])} / {fmt(single['generated_to_nearest_reference_all_atom']['p90'])} / {fmt(single['generated_to_nearest_reference_all_atom']['max'])} | {fmt(single['generated_to_nearest_reference_heavy_atom']['median'])} / {fmt(single['generated_to_nearest_reference_heavy_atom']['p90'])} / {fmt(single['generated_to_nearest_reference_heavy_atom']['max'])} |
| 最终全部 XYZ 到最近参考 | {fmt(summary['final_candidate_nearest_reference_all_atom_rmsd']['median'])} / {fmt(summary['final_candidate_nearest_reference_all_atom_rmsd']['p90'])} / {fmt(summary['final_candidate_nearest_reference_all_atom_rmsd']['max'])} | {fmt(summary['final_candidate_nearest_reference_heavy_atom_rmsd']['median'])} / {fmt(summary['final_candidate_nearest_reference_heavy_atom_rmsd']['p90'])} / {fmt(summary['final_candidate_nearest_reference_heavy_atom_rmsd']['max'])} |
| 最终集合 path-best | {fmt(summary['final_path_best_all_atom_rmsd']['median'])} / {fmt(summary['final_path_best_all_atom_rmsd']['p90'])} / {fmt(summary['final_path_best_all_atom_rmsd']['max'])} | {fmt(summary['final_path_best_heavy_atom_rmsd']['median'])} / {fmt(summary['final_path_best_heavy_atom_rmsd']['p90'])} / {fmt(summary['final_path_best_heavy_atom_rmsd']['max'])} |

## Top-k：生成上限与参考覆盖

- **Oracle path-best**：对每条 path 的前 k 个候选使用真实参考 RMSD，事后挑出最接近任一参考的候选，再对 path 等权平均。它衡量候选池中“有没有好结构”，但真实推理时未知参考结构，因此不可作为筛选器。
- **参考覆盖（reference coverage）**：同一 path 的每个参考构象分别寻找前 k 个候选中的最近结构，先在 path 内对参考求平均，再对 path 等权平均。它衡量生成集合能否覆盖整个参考构象集合，比“至少命中一个”的 path-best 更严格。
- Top1/5/10/50/100 必须在同一候选 sweep 内纵向比较。单候选评估来自另一轮生成，不能要求其数值与 sweep 的 Top1 完全相同。

| 候选数 | Oracle path-best 全原子 | Oracle path-best 重原子 | 参考覆盖全原子 | 参考覆盖重原子 |
|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| Top{k} | {fmt(topk[str(k)]['oracle_path_best_all_atom']['mean'])} | {fmt(topk[str(k)]['oracle_path_best_heavy_atom']['mean'])} | {fmt(topk[str(k)]['reference_coverage_all_atom']['mean'])} | {fmt(topk[str(k)]['reference_coverage_heavy_atom']['mean'])} |"
        for k in (1, 5, 10, 50, 100)
    ) + f"""

Top100 oracle 到实际最终 path-best 的差距为全原子
{summary['final_path_best_all_atom_rmsd']['mean'] - topk['100']['oracle_path_best_all_atom']['mean']:.4f} Å、
重原子 {summary['final_path_best_heavy_atom_rmsd']['mean'] - topk['100']['oracle_path_best_heavy_atom']['mean']:.4f} Å。
这部分差距主要反映“候选生成出来了，但仅靠立体、聚类和 xTB 能量不能总是识别 RMSD 最优候选”，
而不是生成模型本身完全没有产生接近参考的结构。

## 测试集参考构象内部基准

内部基准只在同一 path 的参考构象之间计算，并排除自匹配。{baseline['n_singleton_paths']} 条只有一个参考构象的
path 无法计算最近的“另一个参考”，因此最近邻基准来自 {baseline['n_multiconformer_paths']} 条多构象 path。

| 内部基准定义 | 全原子均值 | 重原子均值 | 聚合方式 |
|---|---:|---:|---|
| 每个参考到同 path 最近的另一个参考 | {fmt(baseline['nearest_other_reference_all_atom_rmsd_micro']['mean'])} | {fmt(baseline['nearest_other_reference_heavy_atom_rmsd_micro']['mean'])} | 对所有可比较参考构象求平均（micro） |
| 每个 path 的上述最近邻均值 | {fmt(baseline['path_mean_nearest_other_all_atom_rmsd_macro']['mean'])} | {fmt(baseline['path_mean_nearest_other_heavy_atom_rmsd_macro']['mean'])} | 对多构象 path 等权平均（macro） |
| 同 path 全部参考两两比较 | {fmt(baseline['all_reference_pairwise_all_atom_rmsd']['mean'])} | {fmt(baseline['all_reference_pairwise_heavy_atom_rmsd']['mean'])} | 对所有参考构象对求平均 |

内部基准不是模型误差阈值，而是数据中已知参考构象的离散尺度。最终 `path-best`
{fmt(summary['final_path_best_all_atom_rmsd']['mean'])}/{fmt(summary['final_path_best_heavy_atom_rmsd']['mean'])} Å
与 path-macro 内部基准
{fmt(baseline['path_mean_nearest_other_all_atom_rmsd_macro']['mean'])}/{fmt(baseline['path_mean_nearest_other_heavy_atom_rmsd_macro']['mean'])} Å
相比，{baseline_interpretation}。二者取最小值和平均的层级仍然不同，不能据此判断每个最终 XYZ 是否达到内部基准。

## 推荐的读法

1. 判断**单次生成稳定性**：看“单候选到同 path 最近参考”，即 {fmt(single['generated_to_nearest_reference_all_atom']['mean'])}/{fmt(single['generated_to_nearest_reference_heavy_atom']['mean'])} Å。
2. 判断**最终交付集合是否至少命中一个可用构象**：看实际“最终集合 path-best”，即 {fmt(summary['final_path_best_all_atom_rmsd']['mean'])}/{fmt(summary['final_path_best_heavy_atom_rmsd']['mean'])} Å。这是当前实际工作流最重要的指标。
3. 判断**所有交付结构的平均质量**：看“最终全部 XYZ 到最近参考”，即 {fmt(summary['final_candidate_nearest_reference_all_atom_rmsd']['mean'])}/{fmt(summary['final_candidate_nearest_reference_heavy_atom_rmsd']['mean'])} Å；它会让输出构象数较多的 path 权重更大。
4. 判断**候选生成器的潜在上限**：看 Top100 oracle；它只能用于诊断候选池，不能作为实际性能宣传值。
5. 判断**是否覆盖已知多构象集合**：看 Top-k reference coverage，而不是 path-best。

## 限制与注意事项

- 参考构象集合有限；“到最近参考 RMSD 较大”既可能是生成错误，也可能是生成了数据集中未收录但合理的构象。
- GFN1-xTB 单点能是快速筛选量，尤其对过渡态并非高精度自由能。这里只在同一 path 内比较相对能量，不跨反应比较绝对能量。
- `{args.energy_threshold:.3f} Eh` 对应 {args.energy_threshold * 627.509474:.3f} kcal/mol，是当前经验能窗；改变能窗会同时改变覆盖率、最终 XYZ 数和 path-best。
- 全原子 RMSD 对氢原子很敏感。判断成键骨架和 TS 主体构象时优先看重原子 RMSD，同时保留全原子结果检查质子和取代基朝向。
- RMSD 只描述坐标相似度，不验证键级、元素顺序或立体符号；本报告分别通过 atom mapping、元素序列核验和立体约束审计补足这些检查。

## 输出文件

- `final_xyz/`：按 path 分目录保存最终多构象 XYZ；每个文件对应一个通过能窗和立体检查的 cluster。
- `final_xyz_manifest.csv`：逐 XYZ 的 path、cluster、候选编号、xTB 能量、相对能量、RMSD、反应 ID、DPA ID 和 mapped reaction SMILES。
- `path_summary.csv`：逐 path 的聚类数、最终 XYZ 数、最低能量和 path-best RMSD。
- `cluster_representatives.csv`：全部立体过滤后 cluster 的类大小、最低能代表及能窗状态。
- `candidate_stereo_energy.csv`：{len(audit_rows)} 个原始候选的完整立体与能量审计；未通过立体检查者的 `stereo_filtered_cluster` 为空。
- `summary.json`：机器可读汇总和分布统计。
"""
    (args.output_dir / "report_zh.md").write_text(report)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
