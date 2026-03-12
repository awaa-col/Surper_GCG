"""
experiments/exp_13_cross_topic_safe_manifold.py
===============================================
跨主题 same-topic contrast 复现实验。

定位：
  - 这是新的 Exp13。
  - 用 train/dev/test 分离方式复验 Exp10 的 late safe-response 线索，
    不再把单主题现象直接外推成普遍规律。
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
    remove_projection,
    split_half_stability,
)
from probes.stats import set_seed


ABL_LAYERS = [17, 23]
LATE_LAYERS = list(range(18, 26))
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


def collect_hidden_states_with_ablation(
    model,
    tokenizer,
    prompts,
    direction,
    layers,
    desc,
):
    with ablation_context(model, direction, layers=ABL_LAYERS):
        return collect_hidden_states(model, tokenizer, prompts, layers=layers, desc=desc)


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
            "std_a": float(a.std(unbiased=False).item()),
            "std_b": float(b.std(unbiased=False).item()),
        }
    return out


def top_layers_by_gap(stats_dict, top_k=5):
    pairs = [(int(layer), abs(info["gap"])) for layer, info in stats_dict.items()]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [layer for layer, _ in pairs[:top_k]]


def cosine_report(directions, ref_dir):
    out = {}
    for layer, vec in directions.items():
        out[str(layer)] = float(torch.dot(vec.to(ref_dir.device), ref_dir).item())
    return out


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="results/exp13_cross_topic_safe_manifold.json",
    )
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
    test_payload = load_topic_banks(split="test", seed=args.seed)

    support_dirs = {}
    safe_info_dirs = {}
    topic_results = {}

    for topic in tqdm(sorted(train_payload.keys()), desc="exp13.topics"):
        train_groups = train_payload[topic]
        dev_groups = dev_payload[topic]
        test_groups = test_payload[topic]

        h_train = collect_hidden_states(
            model,
            tokenizer,
            train_groups["harmful"],
            layers=ALL_LAYERS,
            desc=f"{topic}_harm_train",
        )
        s_train = collect_hidden_states(
            model,
            tokenizer,
            train_groups["supportive"],
            layers=ALL_LAYERS,
            desc=f"{topic}_support_train",
        )
        i_train = collect_hidden_states(
            model,
            tokenizer,
            train_groups["safe_info"],
            layers=ALL_LAYERS,
            desc=f"{topic}_info_train",
        )

        harm_vs_support = mean_diff_direction(h_train, s_train)
        harm_vs_info = mean_diff_direction(h_train, i_train)

        support_stability = split_half_stability(h_train, s_train, k=20, seed=args.seed)
        info_stability = split_half_stability(h_train, i_train, k=20, seed=args.seed)

        h_dev = collect_hidden_states(
            model,
            tokenizer,
            dev_groups["harmful"],
            layers=ALL_LAYERS,
            desc=f"{topic}_harm_dev",
        )
        s_dev = collect_hidden_states(
            model,
            tokenizer,
            dev_groups["supportive"],
            layers=ALL_LAYERS,
            desc=f"{topic}_support_dev",
        )
        i_dev = collect_hidden_states(
            model,
            tokenizer,
            dev_groups["safe_info"],
            layers=ALL_LAYERS,
            desc=f"{topic}_info_dev",
        )

        h_test = collect_hidden_states(
            model,
            tokenizer,
            test_groups["harmful"],
            layers=ALL_LAYERS,
            desc=f"{topic}_harm_test",
        )
        s_test = collect_hidden_states(
            model,
            tokenizer,
            test_groups["supportive"],
            layers=ALL_LAYERS,
            desc=f"{topic}_support_test",
        )
        i_test = collect_hidden_states(
            model,
            tokenizer,
            test_groups["safe_info"],
            layers=ALL_LAYERS,
            desc=f"{topic}_info_test",
        )

        h_train_abl = collect_hidden_states_with_ablation(
            model,
            tokenizer,
            train_groups["harmful"],
            r_exec,
            LATE_LAYERS,
            f"{topic}_harm_train_abl",
        )
        s_train_abl = collect_hidden_states_with_ablation(
            model,
            tokenizer,
            train_groups["supportive"],
            r_exec,
            LATE_LAYERS,
            f"{topic}_support_train_abl",
        )
        i_train_abl = collect_hidden_states_with_ablation(
            model,
            tokenizer,
            train_groups["safe_info"],
            r_exec,
            LATE_LAYERS,
            f"{topic}_info_train_abl",
        )

        support_dir_abl = mean_diff_direction(h_train_abl, s_train_abl)
        info_dir_abl = mean_diff_direction(h_train_abl, i_train_abl)
        support_dir_abl = remove_projection(
            support_dir_abl,
            {layer: r_exec for layer in LATE_LAYERS},
        )
        info_dir_abl = remove_projection(
            info_dir_abl,
            {layer: r_exec for layer in LATE_LAYERS},
        )

        support_dirs[topic] = support_dir_abl
        safe_info_dirs[topic] = info_dir_abl

        support_abl_stability = split_half_stability(
            h_train_abl,
            s_train_abl,
            k=20,
            seed=args.seed,
        )
        info_abl_stability = split_half_stability(
            h_train_abl,
            i_train_abl,
            k=20,
            seed=args.seed,
        )

        h_dev_abl = collect_hidden_states_with_ablation(
            model,
            tokenizer,
            dev_groups["harmful"],
            r_exec,
            LATE_LAYERS,
            f"{topic}_harm_dev_abl",
        )
        s_dev_abl = collect_hidden_states_with_ablation(
            model,
            tokenizer,
            dev_groups["supportive"],
            r_exec,
            LATE_LAYERS,
            f"{topic}_support_dev_abl",
        )
        i_dev_abl = collect_hidden_states_with_ablation(
            model,
            tokenizer,
            dev_groups["safe_info"],
            r_exec,
            LATE_LAYERS,
            f"{topic}_info_dev_abl",
        )

        h_test_abl = collect_hidden_states_with_ablation(
            model,
            tokenizer,
            test_groups["harmful"],
            r_exec,
            LATE_LAYERS,
            f"{topic}_harm_test_abl",
        )
        s_test_abl = collect_hidden_states_with_ablation(
            model,
            tokenizer,
            test_groups["supportive"],
            r_exec,
            LATE_LAYERS,
            f"{topic}_support_test_abl",
        )
        i_test_abl = collect_hidden_states_with_ablation(
            model,
            tokenizer,
            test_groups["safe_info"],
            r_exec,
            LATE_LAYERS,
            f"{topic}_info_test_abl",
        )

        topic_results[topic] = {
            "n_train_per_group": len(train_groups["harmful"]),
            "n_dev_per_group": len(dev_groups["harmful"]),
            "n_test_per_group": len(test_groups["harmful"]),
            "baseline": {
                "harm_vs_supportive": {
                    "train_stability": {
                        str(layer): {
                            "mean": float(support_stability[layer]["mean"]),
                            "std": float(support_stability[layer]["std"]),
                        }
                        for layer in ALL_LAYERS
                    },
                    "train_cosine_with_r_exec": cosine_report(harm_vs_support, r_exec),
                    "dev_gap_stats": layer_stats(h_dev, s_dev, harm_vs_support),
                    "dev_top_layers_by_gap": top_layers_by_gap(
                        layer_stats(h_dev, s_dev, harm_vs_support)
                    ),
                    "test_gap_stats": layer_stats(h_test, s_test, harm_vs_support),
                },
                "harm_vs_safe_info": {
                    "train_stability": {
                        str(layer): {
                            "mean": float(info_stability[layer]["mean"]),
                            "std": float(info_stability[layer]["std"]),
                        }
                        for layer in ALL_LAYERS
                    },
                    "train_cosine_with_r_exec": cosine_report(harm_vs_info, r_exec),
                    "dev_gap_stats": layer_stats(h_dev, i_dev, harm_vs_info),
                    "dev_top_layers_by_gap": top_layers_by_gap(
                        layer_stats(h_dev, i_dev, harm_vs_info)
                    ),
                    "test_gap_stats": layer_stats(h_test, i_test, harm_vs_info),
                },
            },
            "exec_ablation": {
                "harm_vs_supportive": {
                    "train_stability": {
                        str(layer): {
                            "mean": float(support_abl_stability[layer]["mean"]),
                            "std": float(support_abl_stability[layer]["std"]),
                        }
                        for layer in LATE_LAYERS
                    },
                    "train_cosine_with_r_exec_removed": cosine_report(
                        support_dir_abl,
                        r_exec,
                    ),
                    "dev_gap_stats": layer_stats(h_dev_abl, s_dev_abl, support_dir_abl),
                    "dev_top_layers_by_gap": top_layers_by_gap(
                        layer_stats(h_dev_abl, s_dev_abl, support_dir_abl)
                    ),
                    "test_gap_stats": layer_stats(h_test_abl, s_test_abl, support_dir_abl),
                },
                "harm_vs_safe_info": {
                    "train_stability": {
                        str(layer): {
                            "mean": float(info_abl_stability[layer]["mean"]),
                            "std": float(info_abl_stability[layer]["std"]),
                        }
                        for layer in LATE_LAYERS
                    },
                    "train_cosine_with_r_exec_removed": cosine_report(
                        info_dir_abl,
                        r_exec,
                    ),
                    "dev_gap_stats": layer_stats(h_dev_abl, i_dev_abl, info_dir_abl),
                    "dev_top_layers_by_gap": top_layers_by_gap(
                        layer_stats(h_dev_abl, i_dev_abl, info_dir_abl)
                    ),
                    "test_gap_stats": layer_stats(h_test_abl, i_test_abl, info_dir_abl),
                },
            },
        }

    output = {
        "model": args.model,
        "seed": args.seed,
        "target_layers": {
            "exec": ABL_LAYERS,
            "late": LATE_LAYERS,
        },
        "train_prompt_groups": train_payload,
        "dev_prompt_groups": dev_payload,
        "test_prompt_groups": test_payload,
        "topics": topic_results,
        "cross_topic": {
            "supportive_pairwise_cosine": {
                f"L{layer}": pairwise_cosine_matrix(support_dirs, layer)
                for layer in LATE_LAYERS
            },
            "safe_info_pairwise_cosine": {
                f"L{layer}": pairwise_cosine_matrix(safe_info_dirs, layer)
                for layer in LATE_LAYERS
            },
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[exp_13] saved={out_path}")


if __name__ == "__main__":
    main()
