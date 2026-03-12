"""
experiments/exp_17_gemma_scope_feature_probe.py
===============================================
Use Gemma Scope 2 SAEs to probe saved Exp16 responses without regenerating outputs.
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

from data.topic_banks import flatten_topic_group
from probes.gemma_scope import (
    build_scope_release,
    encode_scope_features,
    preload_scope_saes,
    summarize_feature_activations,
    summarize_feature_contrast,
)
from probes.response_spans import build_span_records, collect_segment_hidden_states
from probes.stats import set_seed


DEFAULT_LAYERS = [17, 18, 19, 20, 21, 22, 23, 24, 25]
DEFAULT_CONTRASTS = [
    (
        "harmful_baseline:refusal_clause",
        "harmful_exec_only:unsafe_instructions",
        "refusal_vs_exec_instructions",
    ),
    (
        "harmful_baseline:empathy_apology",
        "harmful_exec_only:unsafe_instructions",
        "empathy_vs_exec_instructions",
    ),
    (
        "harmful_baseline:resource_redirect",
        "harmful_exec_only:unsafe_instructions",
        "redirect_vs_exec_instructions",
    ),
    (
        "harmful_baseline:risk_warning",
        "harmful_exec_only:unsafe_instructions",
        "risk_vs_exec_instructions",
    ),
    (
        "supportive_baseline:supportive_opening",
        "safe_info_baseline:educational_opening",
        "supportive_vs_educational_opening",
    ),
]


def load_model(model_name: str, hf_token: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def max_yes_prob(audit_item: dict) -> float:
    return max(
        float(score_payload.get("yes_prob", 0.0))
        for score_payload in audit_item.get("scores", {}).values()
    )


def select_saved_indices(
    audit_payload: dict,
    *,
    source_group: str,
    safe_max_prob: float,
    unsafe_min_prob: float,
) -> list[int]:
    items = audit_payload.get(source_group, {}).get("items", [])
    if not items:
        return []

    selected = []
    for idx, item in enumerate(items):
        score = max_yes_prob(item)
        if source_group == "harmful_baseline" and score <= safe_max_prob:
            selected.append(idx)
        elif source_group == "harmful_exec_only" and score >= unsafe_min_prob:
            selected.append(idx)
    return selected


def reconstruct_prompts(saved_payload: dict, source_group: str) -> list[str]:
    group_alias = {
        "harmful_baseline": "harmful",
        "harmful_exec_only": "harmful",
        "supportive_baseline": "supportive",
        "safe_info_baseline": "safe_info",
    }[source_group]
    return flatten_topic_group(saved_payload["prompt_groups"], group_alias)


def build_records(saved_payload: dict, args) -> tuple[list[dict], dict]:
    records = []
    selection = {}
    shield_audit = saved_payload.get("shield_audit", {})

    for source_group in [
        "harmful_baseline",
        "harmful_exec_only",
        "supportive_baseline",
        "safe_info_baseline",
    ]:
        prompts = reconstruct_prompts(saved_payload, source_group)
        responses = saved_payload["conditions"][source_group]["responses"]
        indices = list(range(len(responses)))

        if source_group in {"harmful_baseline", "harmful_exec_only"} and shield_audit:
            shield_selected = select_saved_indices(
                shield_audit,
                source_group=source_group,
                safe_max_prob=args.shield_safe_max_prob,
                unsafe_min_prob=args.shield_unsafe_min_prob,
            )
            if shield_selected:
                indices = shield_selected

        selection[source_group] = {
            "available": len(responses),
            "selected": len(indices),
            "indices": indices,
        }

        selected_prompts = [prompts[i] for i in indices]
        selected_responses = [responses[i] for i in indices]
        records.extend(
            build_span_records(
                selected_prompts,
                selected_responses,
                source_group,
            )
        )

    return records, selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument(
        "--input",
        default="results/exp16_safe_response_dictionary_full.json",
    )
    parser.add_argument(
        "--output",
        default="results/exp17_gemma_scope_feature_probe.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scope_site", default="res", choices=["res", "att", "mlp", "transcoder"])
    parser.add_argument("--scope_release", default=None)
    parser.add_argument("--scope_width", default="16k")
    parser.add_argument("--scope_l0", default="small")
    parser.add_argument("--scope_device", default="auto")
    parser.add_argument("--scope_dtype", default="float32")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_group_size", type=int, default=4)
    parser.add_argument("--shield_safe_max_prob", type=float, default=0.35)
    parser.add_argument("--shield_unsafe_min_prob", type=float, default=0.5)
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    args = parser.parse_args()

    set_seed(args.seed)
    release = args.scope_release or build_scope_release(
        args.model,
        site=args.scope_site,
        all_layers=True,
    )
    resolved_scope_device = args.scope_device
    if resolved_scope_device == "auto":
        resolved_scope_device = "cuda" if torch.cuda.is_available() else "cpu"

    scope_saes, scope_infos = preload_scope_saes(
        args.layers,
        release=release,
        width=args.scope_width,
        l0=args.scope_l0,
        device=resolved_scope_device,
        dtype=args.scope_dtype,
        force_download=args.force_download,
    )

    if args.download_only:
        output = {
            "download_only": True,
            "model": args.model,
            "scope_release": release,
            "scope_width": args.scope_width,
            "scope_l0": args.scope_l0,
            "scope_device": resolved_scope_device,
            "scope_infos": {
                str(layer): info.to_dict()
                for layer, info in scope_infos.items()
            },
        }
        Path(args.output).write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[exp_17] saved={args.output}")
        return

    saved_payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    records, selection = build_records(saved_payload, args)

    model, tokenizer = load_model(args.model, args.hf_token)
    group_states, group_metadata = collect_segment_hidden_states(
        model,
        tokenizer,
        records,
        layers=args.layers,
        desc="exp17.span_states",
    )

    valid_groups = {
        group_key: layer_map
        for group_key, layer_map in group_states.items()
        if max(tensor.shape[0] for tensor in layer_map.values()) >= args.min_group_size
    }

    group_feature_summary = {}
    encoded_feature_cache = {}

    total_jobs = sum(len(layer_map) for layer_map in valid_groups.values())
    for group_key in tqdm(valid_groups, desc="exp17.scope_groups"):
        group_feature_summary[group_key] = {
            "n_spans": len(group_metadata.get(group_key, [])),
            "layers": {},
        }
        encoded_feature_cache[group_key] = {}
        for layer, states in valid_groups[group_key].items():
            feature_acts = encode_scope_features(
                scope_saes[layer],
                states,
                batch_size=args.batch_size,
                desc=None if total_jobs > 20 else f"{group_key}.L{layer}",
            )
            encoded_feature_cache[group_key][layer] = feature_acts
            group_feature_summary[group_key]["layers"][str(layer)] = (
                summarize_feature_activations(
                    feature_acts,
                    top_k=args.top_k,
                )
            )

    contrast_summary = {}
    for group_a, group_b, name in DEFAULT_CONTRASTS:
        if group_a not in encoded_feature_cache or group_b not in encoded_feature_cache:
            continue
        contrast_summary[name] = {
            "group_a": group_a,
            "group_b": group_b,
            "layers": {},
        }
        for layer in args.layers:
            if layer not in encoded_feature_cache[group_a]:
                continue
            if layer not in encoded_feature_cache[group_b]:
                continue
            contrast_summary[name]["layers"][str(layer)] = summarize_feature_contrast(
                encoded_feature_cache[group_a][layer],
                encoded_feature_cache[group_b][layer],
                top_k=args.top_k,
            )

    output = {
        "model": args.model,
        "seed": args.seed,
        "input": args.input,
        "scope_release": release,
        "scope_width": args.scope_width,
        "scope_l0": args.scope_l0,
        "scope_device": resolved_scope_device,
        "layers": args.layers,
        "selection": selection,
        "scope_infos": {
            str(layer): info.to_dict()
            for layer, info in scope_infos.items()
        },
        "group_feature_summary": group_feature_summary,
        "contrast_summary": contrast_summary,
    }
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[exp_17] saved={args.output}")


if __name__ == "__main__":
    main()
