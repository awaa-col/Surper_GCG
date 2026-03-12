"""
experiments/exp_16_safe_response_dictionary.py
==============================================
自然安全响应字典提取（第一版）。

目标：
  1. 不再主要依赖 system-prompt persona，而是直接从自然输出里抓晚层响应成分。
  2. 先做 v1：按自然响应片段提取晚层字典，不直接上 SAE。
  3. 对 harmful 输出统一接 ShieldGemma 审计。
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

from data.topic_banks import flatten_topic_group, load_topic_banks
from probes.ablate import generate_with_ablation, generate_normal
from probes.direction_cache import extract_and_cache
from probes.extract import mean_diff_direction, projection_values, split_half_stability
from probes.response_spans import (
    build_span_records,
    collect_segment_hidden_states,
    summarize_span_records,
)
from probes.shield_audit import (
    AuditItem,
    ShieldGemmaAuditor,
    pick_top_audit_candidates,
    select_indices_by_risk,
    summarize_audit_results,
)
from probes.stats import set_seed


ABL_LAYERS = [17, 23]
LATE_LAYERS = list(range(18, 26))


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


def generate_condition(
    model,
    tokenizer,
    prompts,
    *,
    condition: str,
    r_exec: torch.Tensor | None = None,
    max_new_tokens: int = 180,
):
    responses = []
    for prompt in tqdm(prompts, desc=condition):
        if "exec_only" in condition:
            assert r_exec is not None
            response = generate_with_ablation(
                model,
                tokenizer,
                prompt,
                direction=r_exec,
                layers=ABL_LAYERS,
                max_new_tokens=max_new_tokens,
            )
        else:
            response = generate_normal(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
            )
        responses.append(response)
    return responses


def run_shield_audit(
    prompts,
    responses,
    *,
    source_path: str,
    truncate_response: int,
    safe_max_prob: float,
    unsafe_min_prob: float,
):
    auditor = ShieldGemmaAuditor()
    items = [
        AuditItem(
            prompt=prompt,
            response=response,
            source_file="exp16_safe_response_dictionary",
            source_path=source_path,
        )
        for prompt, response in zip(prompts, responses)
    ]
    results = auditor.audit_items(
        items,
        truncate_response=truncate_response,
        progress=f"exp16.shield.{source_path}",
    )
    return {
        "summary": summarize_audit_results(results),
        "top_candidates": pick_top_audit_candidates(results),
        "selection": select_indices_by_risk(
            results,
            safe_max_prob=safe_max_prob,
            unsafe_min_prob=unsafe_min_prob,
        ),
        "items": [result.to_dict() for result in results],
    }


def filter_records_by_indices(records, allowed_indices):
    allowed = set(allowed_indices)
    return [
        record for record in records
        if record["response_index"] in allowed
    ]


def merge_group_states(group_states, keys):
    merged = {}
    for layer in LATE_LAYERS:
        chunks = [
            group_states[key][layer]
            for key in keys
            if key in group_states and layer in group_states[key]
        ]
        if chunks:
            merged[layer] = torch.cat(chunks, dim=0)
    return merged


def layer_gap_stats(states_a, states_b, directions):
    proj_a = projection_values(states_a, directions)
    proj_b = projection_values(states_b, directions)
    stats = {}
    for layer in directions:
        a = proj_a[layer]
        b = proj_b[layer]
        stats[str(layer)] = {
            "mean_a": float(a.mean().item()),
            "mean_b": float(b.mean().item()),
            "gap": float((a.mean() - b.mean()).item()),
            "abs_gap": float(abs(a.mean().item() - b.mean().item())),
            "std_a": float(a.std(unbiased=False).item()),
            "std_b": float(b.std(unbiased=False).item()),
        }
    return stats


def mean_direction_cosine(direction_a, direction_b):
    values = []
    for layer in LATE_LAYERS:
        if layer not in direction_a or layer not in direction_b:
            continue
        values.append(
            float(
                torch.dot(
                    direction_a[layer].to(direction_b[layer].device),
                    direction_b[layer],
                ).item()
            )
        )
    if not values:
        return None
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="all", choices=["train", "dev", "test", "all"])
    parser.add_argument("--n_per_group", type=int, default=8)
    parser.add_argument("--n_train_exec", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--min_group_size", type=int, default=4)
    parser.add_argument("--shield_truncate", type=int, default=500)
    parser.add_argument("--shield_safe_max_prob", type=float, default=0.35)
    parser.add_argument("--shield_unsafe_min_prob", type=float, default=0.5)
    parser.add_argument("--skip_shield_audit", action="store_true")
    parser.add_argument(
        "--output",
        default="results/exp16_safe_response_dictionary.json",
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
        n_train=args.n_train_exec,
        seed=args.seed,
    )[17].to(device)

    topic_payload = load_topic_banks(
        split=args.split,
        seed=args.seed,
        n_per_group=args.n_per_group,
    )
    harmful_prompts = flatten_topic_group(topic_payload, "harmful")
    supportive_prompts = flatten_topic_group(topic_payload, "supportive")
    safe_info_prompts = flatten_topic_group(topic_payload, "safe_info")

    harmful_baseline = generate_condition(
        model,
        tokenizer,
        harmful_prompts,
        condition="exp16.harmful_baseline",
        max_new_tokens=args.max_new_tokens,
    )
    harmful_exec_only = generate_condition(
        model,
        tokenizer,
        harmful_prompts,
        condition="exp16.harmful_exec_only",
        r_exec=r_exec,
        max_new_tokens=args.max_new_tokens,
    )
    supportive_baseline = generate_condition(
        model,
        tokenizer,
        supportive_prompts,
        condition="exp16.supportive_baseline",
        max_new_tokens=args.max_new_tokens,
    )
    safe_info_baseline = generate_condition(
        model,
        tokenizer,
        safe_info_prompts,
        condition="exp16.safe_info_baseline",
        max_new_tokens=args.max_new_tokens,
    )

    shield_audit = {}
    harmful_baseline_selected = list(range(len(harmful_prompts)))
    harmful_exec_only_selected = list(range(len(harmful_prompts)))
    if not args.skip_shield_audit:
        shield_audit["harmful_baseline"] = run_shield_audit(
            harmful_prompts,
            harmful_baseline,
            source_path="harmful_baseline",
            truncate_response=args.shield_truncate,
            safe_max_prob=args.shield_safe_max_prob,
            unsafe_min_prob=args.shield_unsafe_min_prob,
        )
        shield_audit["harmful_exec_only"] = run_shield_audit(
            harmful_prompts,
            harmful_exec_only,
            source_path="harmful_exec_only",
            truncate_response=args.shield_truncate,
            safe_max_prob=args.shield_safe_max_prob,
            unsafe_min_prob=args.shield_unsafe_min_prob,
        )
        harmful_baseline_selected = shield_audit["harmful_baseline"]["selection"][
            "safe_response_indices"
        ]
        harmful_exec_only_selected = shield_audit["harmful_exec_only"]["selection"][
            "unsafe_response_indices"
        ]

    harmful_baseline_records_all = build_span_records(
        harmful_prompts,
        harmful_baseline,
        "harmful_baseline",
    )
    harmful_exec_only_records_all = build_span_records(
        harmful_prompts,
        harmful_exec_only,
        "harmful_exec_only",
    )
    supportive_records_all = build_span_records(
        supportive_prompts,
        supportive_baseline,
        "supportive_baseline",
    )
    safe_info_records_all = build_span_records(
        safe_info_prompts,
        safe_info_baseline,
        "safe_info_baseline",
    )

    span_records_all = []
    span_records_all.extend(harmful_baseline_records_all)
    span_records_all.extend(harmful_exec_only_records_all)
    span_records_all.extend(supportive_records_all)
    span_records_all.extend(safe_info_records_all)

    span_records = []
    span_records.extend(
        filter_records_by_indices(
            harmful_baseline_records_all,
            harmful_baseline_selected,
        )
    )
    span_records.extend(
        filter_records_by_indices(
            harmful_exec_only_records_all,
            harmful_exec_only_selected,
        )
    )
    span_records.extend(supportive_records_all)
    span_records.extend(safe_info_records_all)

    group_states, group_metadata = collect_segment_hidden_states(
        model,
        tokenizer,
        span_records,
        layers=LATE_LAYERS,
        desc="exp16.span_states",
    )

    span_summary_all = summarize_span_records(span_records_all)
    span_summary = summarize_span_records(span_records)

    unsafe_reference = [
        key
        for key in sorted(group_states.keys())
        if key.startswith("harmful_exec_only:")
        and key.split(":", 1)[1] in {"unsafe_instructions", "other"}
    ]

    entry_specs = {
        "supportive_opening": ["supportive_baseline:supportive_opening"],
        "educational_opening": ["safe_info_baseline:educational_opening"],
        "empathy_apology": ["harmful_baseline:empathy_apology"],
        "refusal_clause": ["harmful_baseline:refusal_clause"],
        "risk_warning": ["harmful_baseline:risk_warning"],
        "redirect_clause": ["harmful_baseline:redirect_clause"],
        "resource_redirect": [
            "harmful_baseline:resource_redirect",
            "harmful_baseline:resource_list",
        ],
    }

    dictionary_entries = {}
    entry_dirs = {}
    for name, positive_keys in entry_specs.items():
        positive_states = merge_group_states(group_states, positive_keys)
        negative_states = merge_group_states(group_states, unsafe_reference)
        common_layers = sorted(
            set(positive_states.keys()) & set(negative_states.keys())
        )

        if (
            not positive_states
            or not negative_states
            or not common_layers
            or min(
                positive_states[layer].shape[0] for layer in common_layers
            ) < args.min_group_size
            or min(
                negative_states[layer].shape[0] for layer in common_layers
            ) < args.min_group_size
        ):
            dictionary_entries[name] = {
                "status": "skipped",
                "positive_groups": positive_keys,
                "negative_groups": unsafe_reference,
                "reason": "insufficient_group_size",
            }
            continue

        directions = mean_diff_direction(positive_states, negative_states)
        stability = split_half_stability(
            positive_states,
            negative_states,
            k=20,
            seed=args.seed,
        )
        gaps = layer_gap_stats(positive_states, negative_states, directions)
        entry_dirs[name] = directions

        dictionary_entries[name] = {
            "status": "ok",
            "positive_groups": positive_keys,
            "negative_groups": unsafe_reference,
            "n_positive": {
                str(layer): int(positive_states[layer].shape[0])
                for layer in directions
            },
            "n_negative": {
                str(layer): int(negative_states[layer].shape[0])
                for layer in directions
            },
            "stability": {
                str(layer): {
                    "mean": float(stability[layer]["mean"]),
                    "std": float(stability[layer]["std"]),
                }
                for layer in directions
            },
            "gap_stats": gaps,
            "top_layers_by_gap": sorted(
                (int(layer) for layer in gaps.keys()),
                key=lambda layer: gaps[str(layer)]["abs_gap"],
                reverse=True,
            )[:5],
        }

    cosine_summary = {}
    entry_names = sorted(entry_dirs.keys())
    for name_a in entry_names:
        cosine_summary[name_a] = {}
        for name_b in entry_names:
            if name_a == name_b:
                cosine_summary[name_a][name_b] = 1.0
                continue
            cosine_summary[name_a][name_b] = mean_direction_cosine(
                entry_dirs[name_a],
                entry_dirs[name_b],
            )

    output = {
        "model": args.model,
        "seed": args.seed,
        "split": args.split,
        "n_per_group": args.n_per_group,
        "max_new_tokens": args.max_new_tokens,
        "target_layers": {
            "exec": ABL_LAYERS,
            "late": LATE_LAYERS,
        },
        "shield_filtering": {
            "enabled": not args.skip_shield_audit,
            "safe_max_prob": args.shield_safe_max_prob,
            "unsafe_min_prob": args.shield_unsafe_min_prob,
            "harmful_baseline_selected": harmful_baseline_selected,
            "harmful_exec_only_selected": harmful_exec_only_selected,
        },
        "prompt_groups": topic_payload,
        "conditions": {
            "harmful_baseline": {
                "n": len(harmful_prompts),
                "responses": harmful_baseline,
                "selected_for_dictionary": harmful_baseline_selected,
            },
            "harmful_exec_only": {
                "n": len(harmful_prompts),
                "responses": harmful_exec_only,
                "selected_for_dictionary": harmful_exec_only_selected,
            },
            "supportive_baseline": {
                "n": len(supportive_prompts),
                "responses": supportive_baseline,
            },
            "safe_info_baseline": {
                "n": len(safe_info_prompts),
                "responses": safe_info_baseline,
            },
        },
        "shield_audit": shield_audit,
        "span_records_all": span_records_all,
        "span_records": span_records,
        "span_summary_all": span_summary_all,
        "span_summary": span_summary,
        "span_group_sizes": {
            key: {
                str(layer): int(value[layer].shape[0])
                for layer in value
            }
            for key, value in group_states.items()
        },
        "span_group_examples": {
            key: group_metadata[key][:5] for key in sorted(group_metadata.keys())
        },
        "unsafe_reference_groups": unsafe_reference,
        "dictionary_entry_specs": entry_specs,
        "dictionary_entries": dictionary_entries,
        "dictionary_entry_mean_cosine": cosine_summary,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[exp_16] saved={out_path}")


if __name__ == "__main__":
    main()
