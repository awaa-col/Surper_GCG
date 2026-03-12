"""
experiments/family_structure/exp_32_family_stability.py
=======================================================
Measure whether existing late safety families are stable across:
  1. prompt splits,
  2. topics,
  3. family top-k choices.

This experiment does not write raw risky continuations. It only inspects
prompt-end hidden states with Gemma Scope feature families.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from poc.data.topic_banks import load_topic_banks  # noqa: E402
from poc.experiments.exp_19_l17_l23_late_impact import (  # noqa: E402
    LATE_LAYERS,
    feature_family_scores,
    load_feature_families,
    load_model,
)
from poc.experiments.family_structure.common import (  # noqa: E402
    build_eval_rows,
    parse_name_list,
    resolve_runtime_path,
    save_json,
    stack_capture_rows,
)
from poc.probes.gemma_scope import (  # noqa: E402
    build_scope_release,
    build_scope_sae_id,
    encode_scope_features,
    load_scope_sae,
)
from poc.probes.stats import set_seed  # noqa: E402


DEFAULT_TOPICS = ["self_harm", "explosives", "fraud", "hate_or_abuse"]
DEFAULT_GROUPS = ["harmful", "supportive", "safe_info"]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(value):
        return None
    return value


def split_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        out[row[key]].append(idx)
    return dict(out)


def subset_scores(scores: np.ndarray, indices: list[int]) -> list[float]:
    return [float(scores[idx]) for idx in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topics", default=",".join(DEFAULT_TOPICS))
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--n_per_group", type=int, default=4)
    parser.add_argument("--scope_site", default="res", choices=["res", "att", "mlp", "transcoder"])
    parser.add_argument("--scope_release", default=None)
    parser.add_argument("--scope_width", default="16k")
    parser.add_argument("--scope_l0", default="small")
    parser.add_argument("--scope_device", default="auto")
    parser.add_argument("--scope_dtype", default="float32")
    parser.add_argument("--scope_batch_size", type=int, default=128)
    parser.add_argument("--family_top_ks", default="3,5,8")
    parser.add_argument("--exp17_input", default="results/exp17_gemma_scope_feature_probe_full.json")
    parser.add_argument("--output", default="results/family_structure/exp32_family_stability.json")
    args = parser.parse_args()

    set_seed(args.seed)
    topics = parse_name_list(args.topics)
    groups = parse_name_list(args.groups)
    family_top_ks = [int(item.strip()) for item in args.family_top_ks.split(",") if item.strip()]
    if not family_top_ks:
        raise ValueError("Expected at least one family top-k.")

    model, tokenizer = load_model(args.model, args.hf_token)
    resolved_scope_device = args.scope_device
    if resolved_scope_device == "auto":
        resolved_scope_device = "cuda" if torch.cuda.is_available() else "cpu"
    scope_release = args.scope_release or build_scope_release(args.model, site=args.scope_site, all_layers=True)

    topic_payload = load_topic_banks(
        split=args.split,
        seed=args.seed,
        topics=topics,
        n_per_group=args.n_per_group,
    )
    rows = build_eval_rows(topic_payload, groups=groups)
    prompt_states = stack_capture_rows(model, tokenizer, rows=rows, capture_layers=LATE_LAYERS)
    by_topic = split_rows(rows, "topic")
    by_group = split_rows(rows, "group")

    resolved_exp17 = resolve_runtime_path(args.exp17_input)
    if not resolved_exp17.exists():
        raise FileNotFoundError(f"Missing exp17 input: {resolved_exp17}")

    scope_infos: dict[str, dict] = {}
    family_payloads: dict[str, dict] = {}
    for layer in LATE_LAYERS:
        sae_id = build_scope_sae_id(layer, width=args.scope_width, l0=args.scope_l0)
        sae, info = load_scope_sae(
            release=scope_release,
            sae_id=sae_id,
            device=resolved_scope_device,
            dtype=args.scope_dtype,
        )
        scope_infos[str(layer)] = info.to_dict()
        feature_acts = encode_scope_features(
            sae,
            prompt_states[layer],
            batch_size=args.scope_batch_size,
            desc=f"exp32.layer{layer}",
        )

        per_top_k: dict[str, dict] = {}
        for top_k in family_top_ks:
            feature_families = load_feature_families(
                resolved_exp17,
                layers=LATE_LAYERS,
                top_k=top_k,
            )
            layer_scores = feature_family_scores(
                feature_acts,
                layer=layer,
                feature_families=feature_families,
            )
            family_summary: dict[str, dict] = {}
            for family_name, scores in layer_scores.items():
                family_summary[family_name] = {
                    "global": {
                        "mean": float(np.mean(scores)),
                        "std": float(np.std(scores)),
                    },
                    "by_topic": {
                        topic: {
                            "mean": float(np.mean(subset_scores(scores, indices))),
                            "std": float(np.std(subset_scores(scores, indices))),
                        }
                        for topic, indices in by_topic.items()
                    },
                    "by_group": {
                        group: {
                            "mean": float(np.mean(subset_scores(scores, indices))),
                            "std": float(np.std(subset_scores(scores, indices))),
                        }
                        for group, indices in by_group.items()
                    },
                    "sample_scores": [float(item) for item in scores.tolist()],
                }
            per_top_k[str(top_k)] = family_summary
        family_payloads[str(layer)] = per_top_k
        try:
            sae.to("cpu")
        except Exception:
            pass

    stability: dict[str, dict] = {}
    top_k_names = [str(item) for item in family_top_ks]
    for layer in map(str, LATE_LAYERS):
        stability[layer] = {}
        for top_k_a, top_k_b in combinations(top_k_names, 2):
            pair_key = f"{top_k_a}__vs__{top_k_b}"
            pair_payload: dict[str, dict] = {}
            family_names = sorted(
                set(family_payloads[layer][top_k_a].keys()) & set(family_payloads[layer][top_k_b].keys())
            )
            for family_name in family_names:
                scores_a = np.array(family_payloads[layer][top_k_a][family_name]["sample_scores"], dtype=np.float32)
                scores_b = np.array(family_payloads[layer][top_k_b][family_name]["sample_scores"], dtype=np.float32)
                pair_payload[family_name] = {
                    "pearson": safe_corr(scores_a, scores_b),
                    "mean_gap": float(np.mean(scores_a - scores_b)),
                    "abs_mean_gap": float(np.mean(np.abs(scores_a - scores_b))),
                }
            stability[layer][pair_key] = pair_payload

    payload = {
        "model": args.model,
        "seed": args.seed,
        "split": args.split,
        "topics": topics,
        "groups": groups,
        "n_rows": len(rows),
        "scope_release": scope_release,
        "scope_width": args.scope_width,
        "scope_l0": args.scope_l0,
        "family_top_ks": family_top_ks,
        "scope_infos": scope_infos,
        "rows": rows,
        "family_payloads": family_payloads,
        "stability": stability,
    }
    saved = save_json(payload, args.output)
    print(f"[exp_32] saved={saved}")


if __name__ == "__main__":
    main()
