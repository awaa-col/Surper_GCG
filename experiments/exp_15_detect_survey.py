"""
experiments/exp_15_detect_survey.py
===================================
全层 detect-like signal survey。

定位：
  - 这是新的 Exp15。
  - 在跨主题 same-topic contrast 下，定位 detect-like signal 的自然层位、
    跨主题一致性，以及它与 `r_exec` 的关系。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.topic_banks import load_topic_banks
from probes.ablate import ablation_context
from probes.direction_cache import extract_and_cache
from probes.extract import (
    collect_hidden_states,
    mean_diff_direction,
    projection_values,
    split_half_stability,
)
from probes.stats import set_seed


ABL_LAYERS = [17, 23]
ALL_LAYERS = list(range(26))


def load_model(model_name: str, hf_token: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def collect_hidden_states_with_ablation(model, tokenizer, prompts, direction, desc):
    with ablation_context(model, direction, layers=ABL_LAYERS):
        return collect_hidden_states(model, tokenizer, prompts, layers=ALL_LAYERS, desc=desc)


def layer_stats(states_a, states_b, directions):
    proj_a = projection_values(states_a, directions)
    proj_b = projection_values(states_b, directions)
    out = {}
    for layer in directions:
        a = proj_a[layer]
        b = proj_b[layer]
        out[str(layer)] = {
            "mean_a": float(a.mean().item()),
            "mean_b": float(b.mean().item()),
            "gap": float((a.mean() - b.mean()).item()),
            "abs_gap": float(abs(a.mean().item() - b.mean().item())),
            "std_a": float(a.std(unbiased=False).item()),
            "std_b": float(b.std(unbiased=False).item()),
        }
    return out


def top_layers_by_gap(stats_dict, top_k=5):
    pairs = [(int(layer), info["abs_gap"]) for layer, info in stats_dict.items()]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [layer for layer, _ in pairs[:top_k]]


def pairwise_cosine_matrix(topic_to_dir, layer):
    topics = sorted(topic_to_dir.keys())
    matrix = {}
    for topic_a in topics:
        row = {}
        for topic_b in topics:
            v1 = topic_to_dir[topic_a][layer]
            v2 = topic_to_dir[topic_b][layer].to(v1.device)
            row[topic_b] = float(torch.dot(v1, v2).item())
        matrix[topic_a] = row
    return matrix


def mean_upper_triangle(matrix):
    topics = sorted(matrix.keys())
    values = []
    for idx, topic_a in enumerate(topics):
        for topic_b in topics[idx + 1 :]:
            values.append(matrix[topic_a][topic_b])
    return sum(values) / max(len(values), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability_threshold", type=float, default=0.7)
    parser.add_argument("--cross_topic_threshold", type=float, default=0.4)
    parser.add_argument("--output", default="results/exp15_detect_survey.json")
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model, args.hf_token)
    device = next(model.parameters()).device

    r_exec = extract_and_cache(
        model,
        tokenizer,
        args.model,
        layers=[17],
        n_train=100,
        seed=args.seed,
    )[17].to(device)

    train_payload = load_topic_banks(split="train", seed=args.seed)
    dev_payload = load_topic_banks(split="dev", seed=args.seed)

    baseline_dirs = {}
    ablated_dirs = {}
    per_topic = {}

    for topic in tqdm(sorted(train_payload.keys()), desc="exp15.topics"):
        train_groups = train_payload[topic]
        dev_groups = dev_payload[topic]

        harmful_train = collect_hidden_states(
            model,
            tokenizer,
            train_groups["harmful"],
            layers=ALL_LAYERS,
            desc=f"{topic}_harm_train",
        )
        support_train = collect_hidden_states(
            model,
            tokenizer,
            train_groups["supportive"],
            layers=ALL_LAYERS,
            desc=f"{topic}_support_train",
        )
        safe_train = collect_hidden_states(
            model,
            tokenizer,
            train_groups["safe_info"],
            layers=ALL_LAYERS,
            desc=f"{topic}_safe_train",
        )

        baseline_dir = mean_diff_direction(harmful_train, support_train)
        support_stability = split_half_stability(harmful_train, support_train, k=20, seed=args.seed)
        safe_stability = split_half_stability(harmful_train, safe_train, k=20, seed=args.seed)

        harmful_dev = collect_hidden_states(
            model,
            tokenizer,
            dev_groups["harmful"],
            layers=ALL_LAYERS,
            desc=f"{topic}_harm_dev",
        )
        support_dev = collect_hidden_states(
            model,
            tokenizer,
            dev_groups["supportive"],
            layers=ALL_LAYERS,
            desc=f"{topic}_support_dev",
        )
        safe_dev = collect_hidden_states(
            model,
            tokenizer,
            dev_groups["safe_info"],
            layers=ALL_LAYERS,
            desc=f"{topic}_safe_dev",
        )

        harmful_train_abl = collect_hidden_states_with_ablation(
            model,
            tokenizer,
            train_groups["harmful"],
            r_exec,
            desc=f"{topic}_harm_train_abl",
        )
        support_train_abl = collect_hidden_states_with_ablation(
            model,
            tokenizer,
            train_groups["supportive"],
            r_exec,
            desc=f"{topic}_support_train_abl",
        )
        baseline_dir_abl = mean_diff_direction(harmful_train_abl, support_train_abl)
        ablated_stability = split_half_stability(
            harmful_train_abl,
            support_train_abl,
            k=20,
            seed=args.seed,
        )

        harmful_dev_abl = collect_hidden_states_with_ablation(
            model,
            tokenizer,
            dev_groups["harmful"],
            r_exec,
            desc=f"{topic}_harm_dev_abl",
        )
        support_dev_abl = collect_hidden_states_with_ablation(
            model,
            tokenizer,
            dev_groups["supportive"],
            r_exec,
            desc=f"{topic}_support_dev_abl",
        )

        baseline_dirs[topic] = baseline_dir
        ablated_dirs[topic] = baseline_dir_abl

        per_topic[topic] = {
            "n_train_per_group": len(train_groups["harmful"]),
            "n_dev_per_group": len(dev_groups["harmful"]),
            "baseline": {
                "harm_vs_supportive_stability": {
                    str(layer): {
                        "mean": float(support_stability[layer]["mean"]),
                        "std": float(support_stability[layer]["std"]),
                    }
                    for layer in ALL_LAYERS
                },
                "harm_vs_safe_info_stability": {
                    str(layer): {
                        "mean": float(safe_stability[layer]["mean"]),
                        "std": float(safe_stability[layer]["std"]),
                    }
                    for layer in ALL_LAYERS
                },
                "dev_gap_stats": layer_stats(harmful_dev, support_dev, baseline_dir),
                "dev_top_layers_by_gap": top_layers_by_gap(
                    layer_stats(harmful_dev, support_dev, baseline_dir)
                ),
            },
            "exec_ablation": {
                "harm_vs_supportive_stability": {
                    str(layer): {
                        "mean": float(ablated_stability[layer]["mean"]),
                        "std": float(ablated_stability[layer]["std"]),
                    }
                    for layer in ALL_LAYERS
                },
                "dev_gap_stats": layer_stats(harmful_dev_abl, support_dev_abl, baseline_dir_abl),
                "dev_top_layers_by_gap": top_layers_by_gap(
                    layer_stats(harmful_dev_abl, support_dev_abl, baseline_dir_abl)
                ),
            },
            "baseline_vs_ablation_cosine": {
                str(layer): float(
                    torch.dot(
                        baseline_dir[layer].to(baseline_dir_abl[layer].device),
                        baseline_dir_abl[layer],
                    ).item()
                )
                for layer in ALL_LAYERS
            },
        }

    mean_stability_by_layer = {}
    mean_gap_by_layer = {}
    mean_cross_topic_cosine_by_layer = {}
    mean_baseline_vs_ablation_cosine_by_layer = {}

    pairwise_by_layer = {}
    for layer in ALL_LAYERS:
        layer_key = str(layer)
        pairwise_matrix = pairwise_cosine_matrix(baseline_dirs, layer)
        pairwise_by_layer[layer_key] = pairwise_matrix

        mean_stability_by_layer[layer_key] = sum(
            per_topic[topic]["baseline"]["harm_vs_supportive_stability"][layer_key]["mean"]
            for topic in per_topic
        ) / len(per_topic)
        mean_gap_by_layer[layer_key] = sum(
            per_topic[topic]["baseline"]["dev_gap_stats"][layer_key]["abs_gap"]
            for topic in per_topic
        ) / len(per_topic)
        mean_cross_topic_cosine_by_layer[layer_key] = mean_upper_triangle(pairwise_matrix)
        mean_baseline_vs_ablation_cosine_by_layer[layer_key] = sum(
            per_topic[topic]["baseline_vs_ablation_cosine"][layer_key]
            for topic in per_topic
        ) / len(per_topic)

    earliest_consistent_layer = None
    for layer in ALL_LAYERS:
        layer_key = str(layer)
        if (
            mean_stability_by_layer[layer_key] >= args.stability_threshold
            and mean_cross_topic_cosine_by_layer[layer_key] >= args.cross_topic_threshold
        ):
            earliest_consistent_layer = layer
            break

    output = {
        "model": args.model,
        "seed": args.seed,
        "target_layers": {
            "exec": ABL_LAYERS,
            "survey": ALL_LAYERS,
        },
        "train_prompt_groups": train_payload,
        "dev_prompt_groups": dev_payload,
        "topics": per_topic,
        "aggregate": {
            "mean_stability_by_layer": mean_stability_by_layer,
            "mean_abs_dev_gap_by_layer": mean_gap_by_layer,
            "mean_cross_topic_cosine_by_layer": mean_cross_topic_cosine_by_layer,
            "mean_baseline_vs_ablation_cosine_by_layer": mean_baseline_vs_ablation_cosine_by_layer,
            "earliest_consistent_layer": earliest_consistent_layer,
            "pairwise_cosine_by_layer": pairwise_by_layer,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[exp_15] saved={out_path}")


if __name__ == "__main__":
    main()
