"""
experiments/causal_topology/exp_37_boundary_lineage.py
======================================================
Compare whether `soft_apology`, `disclaimer_dangerous_content`, and `direct_unsafe`
look like:
  1. distinct internal states, or
  2. progressively stronger points on one boundary-state lineage.

This script consumes the output of Exp36 and computes:
  - endpoint geometry distances,
  - transition timing distances,
  - trajectory-label overlap,
  - a simple nearest-neighbor lineage summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.family_structure.common import resolve_runtime_path, save_json  # noqa: E402


DEFAULT_SAMPLES = [
    "soft_apology",
    "disclaimer_dangerous_content",
    "direct_unsafe",
]
DEFAULT_CONDITIONS = ["baseline", "l23_only", "l17_l23"]


def parse_name_list(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one name.")
    return items


def euclidean(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a.keys()) | set(b.keys()))
    return math.sqrt(sum((float(a.get(key, 0.0)) - float(b.get(key, 0.0))) ** 2 for key in keys))


def endpoint_vector(summary: dict[str, Any]) -> dict[str, float]:
    late = ((summary.get("final_late_family_activation") or {}).get("23") or {})
    vector = {
        "z_exec": float(summary.get("final_z_exec") or 0.0),
        "z_l23": float(summary.get("final_z_l23") or 0.0),
        "z_soft": float(summary.get("final_z_soft_safe_style_candidate") or 0.0),
    }
    for family_name, value in late.items():
        vector[f"late23:{family_name}"] = float(value)
    return vector


def transition_signature(summary: dict[str, Any]) -> dict[str, float]:
    first = summary.get("first_prefix_by_label") or {}
    return {
        "first_hard_refusal_safe": float(first.get("hard_refusal_safe", -1)),
        "first_soft_safe_shell": float(first.get("soft_safe_shell", -1)),
        "first_disclaimer_danger": float(first.get("disclaimer_danger", -1)),
        "first_unsafe_exec": float(first.get("unsafe_exec", -1)),
        "risk_cross_0_5": float(summary.get("shield_risk_prefix_over_0_5", -1) or -1),
        "risk_cross_0_8": float(summary.get("shield_risk_prefix_over_0_8", -1) or -1),
        "lock_0_5": float(summary.get("danger_token_lock_prefix_over_0_5", -1) or -1),
        "lock_0_9": float(summary.get("danger_token_lock_prefix_over_0_9", -1) or -1),
    }


def normalize_counts(counts: dict[str, int]) -> dict[str, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {key: float(value) / total for key, value in counts.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp36_input",
        default="results/causal_topology/exp36_boundary_profile_full.json",
    )
    parser.add_argument("--sample_names", default=",".join(DEFAULT_SAMPLES))
    parser.add_argument("--condition_names", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--output", default="results/causal_topology/exp37_boundary_lineage.json")
    args = parser.parse_args()

    sample_names = parse_name_list(args.sample_names)
    condition_names = parse_name_list(args.condition_names)
    exp36_input = resolve_runtime_path(args.exp36_input)
    payload = json.loads(exp36_input.read_text(encoding="utf-8"))
    comparison = payload["comparison_summary"]

    pairwise: dict[str, Any] = {}
    nearest: dict[str, Any] = {}

    for condition_name in condition_names:
        condition_pairs: dict[str, Any] = {}
        for i, src_name in enumerate(sample_names):
            src_summary = comparison[src_name][condition_name]
            src_endpoint = endpoint_vector(src_summary)
            src_transition = transition_signature(src_summary)
            src_counts = normalize_counts(src_summary.get("counts") or {})

            best_neighbor = None
            for dst_name in sample_names[i + 1 :]:
                dst_summary = comparison[dst_name][condition_name]
                dst_endpoint = endpoint_vector(dst_summary)
                dst_transition = transition_signature(dst_summary)
                dst_counts = normalize_counts(dst_summary.get("counts") or {})

                endpoint_distance = euclidean(src_endpoint, dst_endpoint)
                transition_distance = euclidean(src_transition, dst_transition)
                count_distance = euclidean(src_counts, dst_counts)
                pair_key = f"{src_name}__vs__{dst_name}"
                condition_pairs[pair_key] = {
                    "endpoint_distance": endpoint_distance,
                    "transition_distance": transition_distance,
                    "count_distance": count_distance,
                    "src_final_label": src_summary.get("final_trajectory_label"),
                    "dst_final_label": dst_summary.get("final_trajectory_label"),
                }

            for dst_name in sample_names:
                if dst_name == src_name:
                    continue
                pair_key = "__vs__".join(sorted([src_name, dst_name]))
                pair = condition_pairs.get(pair_key)
                if pair is None:
                    continue
                score = pair["endpoint_distance"] + pair["transition_distance"] + pair["count_distance"]
                if best_neighbor is None or score < best_neighbor["score"]:
                    best_neighbor = {
                        "neighbor": dst_name,
                        "score": score,
                        "endpoint_distance": pair["endpoint_distance"],
                        "transition_distance": pair["transition_distance"],
                        "count_distance": pair["count_distance"],
                    }
            nearest[f"{condition_name}:{src_name}"] = best_neighbor

        pairwise[condition_name] = condition_pairs

    result = {
        "exp36_input": str(exp36_input),
        "sample_names": sample_names,
        "condition_names": condition_names,
        "pairwise": pairwise,
        "nearest_neighbor_lineage": nearest,
    }
    output_path = save_json(result, args.output)
    print(f"[exp_37] saved={output_path}")


if __name__ == "__main__":
    main()
