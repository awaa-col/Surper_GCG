"""
experiments/exp_40_gate_discovery.py
====================================
12B-first gate discovery bootstrap.

This script prepares the first real gate-discovery pass without importing the
historical L17/L23 prior:

1. extract one candidate direction per scanned layer,
2. ablate each layer with its own direction,
3. compare risk-release gain against baseline,
4. filter out layers that mainly create incoherence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import load_default_datasets
from probes.ablate import generate_normal, generate_with_ablation
from probes.extract import collect_hidden_states, mean_diff_direction, split_half_stability
from probes.model_config import get_hidden_size, get_num_hidden_layers
from probes.stats import batch_classify, set_seed


def load_model(model_name: str, hf_token: str | None = None):
    print(f"[exp_40] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    num_layers = get_num_hidden_layers(model.config)
    hidden_size = get_hidden_size(model.config)
    print(f"  layers={num_layers}, hidden={hidden_size}")
    return model, tokenizer, num_layers


def parse_layer_list(raw: str | None, num_layers: int) -> list[int]:
    if not raw:
        return list(range(num_layers))
    layers = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    invalid = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if invalid:
        raise ValueError(f"Invalid layer indices: {invalid}")
    return layers


def compute_rate(stats: dict, key: str) -> float:
    n = max(int(stats["n"]), 1)
    return float(stats[key]) / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--output", default="results/exp40_gate_discovery.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=32)
    parser.add_argument("--n_eval", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument(
        "--layers",
        default="",
        help="Comma-separated layer list. Empty means scan every layer.",
    )
    parser.add_argument(
        "--min_stability",
        type=float,
        default=0.85,
        help="Flag layers below this split-half stability as weak candidates.",
    )
    parser.add_argument(
        "--max_incoherent_rate",
        type=float,
        default=0.25,
        help="Flag layers above this incoherence rate as destructive.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="How many gate candidates to highlight in the ranked summary.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer, num_layers = load_model(args.model, args.hf_token)
    scan_layers = parse_layer_list(args.layers, num_layers)

    harmful_train, harmless_train = load_default_datasets(
        n_harmful=args.n_train,
        n_harmless=args.n_train,
        split="train",
        seed=args.seed,
    )
    harmful_dev, _ = load_default_datasets(
        n_harmful=args.n_eval,
        n_harmless=args.n_eval,
        split="dev",
        seed=args.seed,
    )

    print(f"[exp_40] train harmful={len(harmful_train)} harmless={len(harmless_train)}")
    print(f"[exp_40] dev harmful={len(harmful_dev)} scan_layers={len(scan_layers)}")

    harmful_states = collect_hidden_states(
        model,
        tokenizer,
        harmful_train,
        layers=scan_layers,
        desc="exp40.train.harmful",
    )
    harmless_states = collect_hidden_states(
        model,
        tokenizer,
        harmless_train,
        layers=scan_layers,
        desc="exp40.train.harmless",
    )
    directions = mean_diff_direction(harmful_states, harmless_states)
    stability = split_half_stability(
        harmful_states,
        harmless_states,
        k=20,
        seed=args.seed,
    )

    baseline_responses = [
        generate_normal(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
        )
        for prompt in tqdm(harmful_dev, desc="exp40.baseline", leave=False)
    ]
    baseline_stats = batch_classify(
        baseline_responses,
        harmful_dev,
        source_path="exp40.baseline",
    )

    baseline_risk_rate = compute_rate(baseline_stats, "success")
    baseline_refusal_rate = compute_rate(baseline_stats, "refused")
    baseline_incoherent_rate = compute_rate(baseline_stats, "incoherent")

    layer_results: dict[str, dict] = {}
    for layer in scan_layers:
        layer_responses = [
            generate_with_ablation(
                model,
                tokenizer,
                prompt,
                directions[layer],
                layers=[layer],
                max_new_tokens=args.max_new_tokens,
            )
            for prompt in tqdm(harmful_dev, desc=f"exp40.layer{layer}", leave=False)
        ]
        stats = batch_classify(
            layer_responses,
            harmful_dev,
            source_path=f"exp40.layer_{layer}",
        )
        risk_rate = compute_rate(stats, "success")
        refusal_rate = compute_rate(stats, "refused")
        incoherent_rate = compute_rate(stats, "incoherent")
        release_gain = risk_rate - baseline_risk_rate
        refusal_drop = baseline_refusal_rate - refusal_rate
        destructive = incoherent_rate > args.max_incoherent_rate
        stable_enough = float(stability[layer]["mean"]) >= args.min_stability

        layer_results[str(layer)] = {
            "layer": layer,
            "direction_norm": float(directions[layer].norm().item()),
            "stability": {
                "mean": float(stability[layer]["mean"]),
                "std": float(stability[layer]["std"]),
            },
            "baseline": {
                "risk_release_rate": baseline_risk_rate,
                "refusal_rate": baseline_refusal_rate,
                "incoherent_rate": baseline_incoherent_rate,
            },
            "ablated": {
                "risk_release_rate": risk_rate,
                "refusal_rate": refusal_rate,
                "incoherent_rate": incoherent_rate,
                "stats": stats,
            },
            "deltas_vs_baseline": {
                "risk_release_gain": release_gain,
                "refusal_drop": refusal_drop,
                "incoherent_increase": incoherent_rate - baseline_incoherent_rate,
            },
            "flags": {
                "stable_enough": stable_enough,
                "destructive": destructive,
                "candidate_gate": stable_enough and (not destructive) and release_gain > 0.0,
            },
            "sample_responses": [
                {
                    "prompt": harmful_dev[idx],
                    "response": layer_responses[idx][:300],
                }
                for idx in range(min(3, len(layer_responses)))
            ],
        }

    ranked_layers = sorted(
        layer_results.values(),
        key=lambda item: (
            not item["flags"]["candidate_gate"],
            -item["deltas_vs_baseline"]["risk_release_gain"],
            item["ablated"]["incoherent_rate"],
            -item["stability"]["mean"],
        ),
    )
    ranked_summary = [
        {
            "layer": item["layer"],
            "candidate_gate": item["flags"]["candidate_gate"],
            "stable_enough": item["flags"]["stable_enough"],
            "destructive": item["flags"]["destructive"],
            "risk_release_gain": round(item["deltas_vs_baseline"]["risk_release_gain"], 4),
            "refusal_drop": round(item["deltas_vs_baseline"]["refusal_drop"], 4),
            "incoherent_rate": round(item["ablated"]["incoherent_rate"], 4),
            "stability_mean": round(item["stability"]["mean"], 4),
        }
        for item in ranked_layers[: args.top_k]
    ]

    output = {
        "model": args.model,
        "seed": args.seed,
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "max_new_tokens": args.max_new_tokens,
        "scan_layers": scan_layers,
        "thresholds": {
            "min_stability": args.min_stability,
            "max_incoherent_rate": args.max_incoherent_rate,
        },
        "baseline_summary": {
            "risk_release_rate": baseline_risk_rate,
            "refusal_rate": baseline_refusal_rate,
            "incoherent_rate": baseline_incoherent_rate,
            "stats": baseline_stats,
        },
        "ranked_candidates": ranked_summary,
        "layer_results": layer_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[exp_40] saved={output_path}")


if __name__ == "__main__":
    main()
