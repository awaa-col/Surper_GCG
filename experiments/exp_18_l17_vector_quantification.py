"""
experiments/exp_18_l17_vector_quantification.py
===============================================
Quantify the influence of the two L17 vectors:
  - r_exec: refusal execution direction
  - r_detect: harm detection direction extracted under r_exec ablation

This experiment is designed to be ShieldGemma-first without regenerating model
outputs. It reuses saved Exp16 prompts, responses, ShieldGemma audit scores,
and span labels, then aligns them with prompt-time L17 projections.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import load_default_datasets
from data.topic_banks import flatten_topic_group
from probes.ablate import _make_ablate_hook, ablation_context
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
TARGET_LAYER = 17
BASELINE_SAFE_RESPONSE_GROUP = "harmful_baseline"
EXEC_ABLATED_RESPONSE_GROUP = "harmful_exec_only"
BASELINE_SPAN_LABELS = [
    "refusal_clause",
    "risk_warning",
    "empathy_apology",
    "redirect_clause",
    "resource_redirect",
    "resource_list",
]
EXEC_SPAN_LABELS = [
    "unsafe_instructions",
    "refusal_clause",
    "risk_warning",
    "empathy_apology",
    "redirect_clause",
    "resource_redirect",
    "resource_list",
]


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


def flatten_prompt_groups_with_topics(
    prompt_groups: Dict[str, Dict[str, List[str]]],
    group_name: str,
) -> List[dict]:
    rows: List[dict] = []
    for topic in sorted(prompt_groups.keys()):
        for prompt in prompt_groups[topic][group_name]:
            rows.append({"topic": topic, "prompt": prompt})
    return rows


def collect_states_with_optional_ablation(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    direction: torch.Tensor | None = None,
    abl_layers: Sequence[int] | None = None,
    target_layer: int = TARGET_LAYER,
    desc: str,
) -> torch.Tensor:
    if direction is None:
        states = collect_hidden_states(
            model,
            tokenizer,
            list(prompts),
            layers=[target_layer],
            desc=desc,
        )
        return states[target_layer]

    captured: list[torch.Tensor] = []
    device = next(model.parameters()).device
    target_module = model.model.layers[target_layer]

    for prompt in tqdm(list(prompts), desc=desc):
        from probes.extract import _build_prompt

        text = _build_prompt(tokenizer, prompt)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        state_holder: dict[str, torch.Tensor] = {}
        hooks = []
        try:
            for layer_idx in list(abl_layers or ABL_LAYERS):
                hooks.append(
                    model.model.layers[layer_idx].register_forward_hook(
                        _make_ablate_hook(direction)
                    )
                )

            def capture_hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                state_holder["state"] = hidden[0, -1, :].float().cpu()
                return output

            hooks.append(target_module.register_forward_hook(capture_hook))

            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            for handle in hooks:
                handle.remove()

        if "state" not in state_holder:
            raise RuntimeError(
                f"Failed to capture post-hook hidden state for layer {target_layer}."
            )
        captured.append(state_holder["state"])

    return torch.stack(captured, dim=0)


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def safe_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b).item())


def max_yes_prob(audit_item: dict) -> float:
    return max(
        float(score_payload.get("yes_prob", 0.0))
        for score_payload in audit_item.get("scores", {}).values()
    )


def dangerous_yes_prob(audit_item: dict) -> float:
    return float(audit_item.get("scores", {}).get("dangerous", {}).get("yes_prob", 0.0))


def projection_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def cohen_d(values_pos: np.ndarray, values_neg: np.ndarray) -> float | None:
    if len(values_pos) == 0 or len(values_neg) == 0:
        return None
    mean_gap = float(np.mean(values_pos) - np.mean(values_neg))
    var_pos = float(np.var(values_pos, ddof=0))
    var_neg = float(np.var(values_neg, ddof=0))
    pooled = ((len(values_pos) * var_pos) + (len(values_neg) * var_neg)) / max(
        len(values_pos) + len(values_neg),
        1,
    )
    if pooled <= 0:
        return None
    return mean_gap / math.sqrt(pooled)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sorter = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[sorter]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[sorter[start:end]] = avg_rank
        start = end
    return ranks


def auroc_from_binary_labels(scores: np.ndarray, labels: np.ndarray) -> float | None:
    labels = labels.astype(np.int64)
    pos = int(labels.sum())
    neg = int((1 - labels).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = rankdata_average(scores)
    rank_sum_pos = float(ranks[labels == 1].sum())
    u = rank_sum_pos - (pos * (pos + 1) / 2.0)
    return float(u / (pos * neg))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return None
    corr = float(np.corrcoef(x, y)[0, 1])
    return None if math.isnan(corr) else corr


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    return pearson_corr(np.asarray(rx, dtype=np.float64), np.asarray(ry, dtype=np.float64))


def linear_r2(features: np.ndarray, target: np.ndarray) -> dict[str, object]:
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if len(target) < 2:
        return {"r2": None, "coefficients": [], "intercept": None}

    design = np.concatenate(
        [np.ones((features.shape[0], 1), dtype=np.float64), features.astype(np.float64)],
        axis=1,
    )
    coeffs, _, _, _ = np.linalg.lstsq(design, target.astype(np.float64), rcond=None)
    preds = design @ coeffs
    ss_res = float(np.sum((target - preds) ** 2))
    ss_tot = float(np.sum((target - float(np.mean(target))) ** 2))
    r2 = None if ss_tot <= 1e-12 else float(1.0 - (ss_res / ss_tot))
    return {
        "r2": r2,
        "intercept": float(coeffs[0]),
        "coefficients": [float(value) for value in coeffs[1:]],
    }


def binary_score_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return {
            "n_positive": int(labels.sum()),
            "n_negative": int((1 - labels).sum()),
            "mean_positive": None,
            "mean_negative": None,
            "mean_gap": None,
            "auroc": None,
            "cohen_d": None,
        }
    return {
        "n_positive": int(labels.sum()),
        "n_negative": int((1 - labels).sum()),
        "mean_positive": float(np.mean(positives)),
        "mean_negative": float(np.mean(negatives)),
        "mean_gap": float(np.mean(positives) - np.mean(negatives)),
        "auroc": auroc_from_binary_labels(scores, labels),
        "cohen_d": cohen_d(positives, negatives),
    }


def continuous_score_metrics(scores: np.ndarray, target: np.ndarray) -> dict[str, object]:
    return {
        "pearson_r": pearson_corr(scores, target),
        "spearman_r": spearman_corr(scores, target),
        "linear_fit": linear_r2(scores, target),
    }


def summarize_direction_on_split(
    values_harmful: np.ndarray,
    values_harmless: np.ndarray,
) -> dict[str, object]:
    labels = np.concatenate(
        [
            np.ones(len(values_harmful), dtype=np.int64),
            np.zeros(len(values_harmless), dtype=np.int64),
        ]
    )
    scores = np.concatenate([values_harmful, values_harmless])
    return {
        "harmful_summary": projection_summary(values_harmful),
        "harmless_summary": projection_summary(values_harmless),
        "mean_gap": float(np.mean(values_harmful) - np.mean(values_harmless)),
        "auroc": auroc_from_binary_labels(scores, labels),
        "cohen_d": cohen_d(values_harmful, values_harmless),
    }


def span_presence_by_response(saved_payload: dict, source_group: str) -> dict[int, set[str]]:
    response_labels: dict[int, set[str]] = defaultdict(set)
    for record in saved_payload.get("span_records", []):
        if record.get("source_group") != source_group:
            continue
        response_index = int(record["response_index"])
        for span in record.get("spans", []):
            label = span.get("label")
            if label:
                response_labels[response_index].add(label)
    return response_labels


def build_joint_metrics(records: Sequence[dict], target_key: str) -> dict[str, object]:
    target = np.asarray([record[target_key] for record in records], dtype=np.float64)
    z_exec_baseline = np.asarray(
        [record["z_exec_baseline"] for record in records],
        dtype=np.float64,
    )
    z_detect_baseline = np.asarray(
        [record["z_detect_baseline"] for record in records],
        dtype=np.float64,
    )
    z_exec_ablated = np.asarray(
        [record["z_exec_ablated"] for record in records],
        dtype=np.float64,
    )
    z_detect_ablated = np.asarray(
        [record["z_detect_ablated"] for record in records],
        dtype=np.float64,
    )
    return {
        "baseline_pair": linear_r2(
            np.stack([z_exec_baseline, z_detect_baseline], axis=1),
            target,
        ),
        "baseline_pair_plus_interaction": linear_r2(
            np.stack(
                [
                    z_exec_baseline,
                    z_detect_baseline,
                    z_exec_baseline * z_detect_baseline,
                ],
                axis=1,
            ),
            target,
        ),
        "ablated_pair": linear_r2(
            np.stack([z_exec_ablated, z_detect_ablated], axis=1),
            target,
        ),
        "ablated_pair_plus_interaction": linear_r2(
            np.stack(
                [
                    z_exec_ablated,
                    z_detect_ablated,
                    z_exec_ablated * z_detect_ablated,
                ],
                axis=1,
            ),
            target,
        ),
    }


def topic_binary_metrics(records: Sequence[dict], key: str) -> dict[str, object]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in records:
        grouped[record["topic"]].append(int(record[key]))
    return {
        topic: {
            "n": len(values),
            "rate": float(sum(values) / len(values)),
        }
        for topic, values in sorted(grouped.items())
    }


def build_per_prompt_records(
    prompt_rows: Sequence[dict],
    baseline_proj_exec: np.ndarray,
    baseline_proj_detect: np.ndarray,
    ablated_proj_exec: np.ndarray,
    ablated_proj_detect: np.ndarray,
    baseline_audit_items: Sequence[dict],
    exec_audit_items: Sequence[dict],
    baseline_span_presence: dict[int, set[str]],
    exec_span_presence: dict[int, set[str]],
) -> list[dict]:
    records: list[dict] = []
    for idx, row in enumerate(prompt_rows):
        baseline_item = baseline_audit_items[idx]
        exec_item = exec_audit_items[idx]
        baseline_labels = baseline_span_presence.get(idx, set())
        exec_labels = exec_span_presence.get(idx, set())

        record = {
            "index": idx,
            "topic": row["topic"],
            "prompt": row["prompt"],
            "z_exec_baseline": float(baseline_proj_exec[idx]),
            "z_detect_baseline": float(baseline_proj_detect[idx]),
            "z_exec_ablated": float(ablated_proj_exec[idx]),
            "z_detect_ablated": float(ablated_proj_detect[idx]),
            "baseline_max_yes_prob": max_yes_prob(baseline_item),
            "exec_only_max_yes_prob": max_yes_prob(exec_item),
            "baseline_dangerous_yes_prob": dangerous_yes_prob(baseline_item),
            "exec_only_dangerous_yes_prob": dangerous_yes_prob(exec_item),
            "risk_delta_max_yes_prob": max_yes_prob(exec_item) - max_yes_prob(baseline_item),
            "risk_delta_dangerous_yes_prob": dangerous_yes_prob(exec_item) - dangerous_yes_prob(baseline_item),
            "baseline_over_0_5": int(max_yes_prob(baseline_item) >= 0.5),
            "exec_only_over_0_5": int(max_yes_prob(exec_item) >= 0.5),
            "exec_only_over_0_8": int(max_yes_prob(exec_item) >= 0.8),
            "baseline_span_labels": sorted(baseline_labels),
            "exec_only_span_labels": sorted(exec_labels),
        }
        for label in BASELINE_SPAN_LABELS:
            record[f"baseline_has_{label}"] = int(label in baseline_labels)
        for label in EXEC_SPAN_LABELS:
            record[f"exec_only_has_{label}"] = int(label in exec_labels)
        records.append(record)
    return records


def summarize_saved_prompt_metrics(records: Sequence[dict]) -> dict[str, object]:
    metrics: dict[str, object] = {
        "n_prompts": len(records),
        "topic_counts": dict(Counter(record["topic"] for record in records)),
        "outcome_rates_by_topic": {
            "baseline_over_0_5": topic_binary_metrics(records, "baseline_over_0_5"),
            "exec_only_over_0_5": topic_binary_metrics(records, "exec_only_over_0_5"),
            "exec_only_over_0_8": topic_binary_metrics(records, "exec_only_over_0_8"),
        },
        "continuous_targets": {},
        "binary_targets": {},
        "span_targets": {
            BASELINE_SAFE_RESPONSE_GROUP: {},
            EXEC_ABLATED_RESPONSE_GROUP: {},
        },
    }

    score_columns = {
        "z_exec_baseline": np.asarray([record["z_exec_baseline"] for record in records], dtype=np.float64),
        "z_detect_baseline": np.asarray([record["z_detect_baseline"] for record in records], dtype=np.float64),
        "z_exec_ablated": np.asarray([record["z_exec_ablated"] for record in records], dtype=np.float64),
        "z_detect_ablated": np.asarray([record["z_detect_ablated"] for record in records], dtype=np.float64),
    }

    continuous_targets = [
        "baseline_max_yes_prob",
        "exec_only_max_yes_prob",
        "baseline_dangerous_yes_prob",
        "exec_only_dangerous_yes_prob",
        "risk_delta_max_yes_prob",
        "risk_delta_dangerous_yes_prob",
    ]
    for target_key in continuous_targets:
        target = np.asarray([record[target_key] for record in records], dtype=np.float64)
        metrics["continuous_targets"][target_key] = {
            score_name: continuous_score_metrics(scores, target)
            for score_name, scores in score_columns.items()
        }
        metrics["continuous_targets"][target_key]["joint_models"] = build_joint_metrics(
            records,
            target_key,
        )

    binary_targets = [
        "baseline_over_0_5",
        "exec_only_over_0_5",
        "exec_only_over_0_8",
    ]
    for target_key in binary_targets:
        labels = np.asarray([record[target_key] for record in records], dtype=np.int64)
        metrics["binary_targets"][target_key] = {
            score_name: binary_score_metrics(scores, labels)
            for score_name, scores in score_columns.items()
        }

    for label in BASELINE_SPAN_LABELS:
        target_key = f"baseline_has_{label}"
        labels = np.asarray([record[target_key] for record in records], dtype=np.int64)
        metrics["span_targets"][BASELINE_SAFE_RESPONSE_GROUP][label] = {
            "rate": float(labels.mean()),
            "z_exec_baseline": binary_score_metrics(score_columns["z_exec_baseline"], labels),
            "z_detect_baseline": binary_score_metrics(score_columns["z_detect_baseline"], labels),
        }

    for label in EXEC_SPAN_LABELS:
        target_key = f"exec_only_has_{label}"
        labels = np.asarray([record[target_key] for record in records], dtype=np.int64)
        metrics["span_targets"][EXEC_ABLATED_RESPONSE_GROUP][label] = {
            "rate": float(labels.mean()),
            "z_exec_ablated": binary_score_metrics(score_columns["z_exec_ablated"], labels),
            "z_detect_ablated": binary_score_metrics(score_columns["z_detect_ablated"], labels),
        }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_eval_harmful", type=int, default=36)
    parser.add_argument(
        "--exp16_input",
        default="results/exp16_safe_response_dictionary_full.json",
    )
    parser.add_argument(
        "--output",
        default="results/exp18_l17_vector_quantification.json",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model, args.hf_token)
    device = next(model.parameters()).device

    exp16_payload = json.loads(Path(args.exp16_input).read_text(encoding="utf-8"))

    # Part A: extract/load the two L17 directions and quantify train-set geometry.
    r_exec = extract_and_cache(
        model,
        tokenizer,
        args.model,
        layers=[TARGET_LAYER],
        n_train=args.n_train,
        seed=args.seed,
    )[TARGET_LAYER].to(device)

    harmful_train, harmless_train = load_default_datasets(
        n_harmful=args.n_train,
        n_harmless=args.n_train,
        split="train",
        seed=args.seed,
    )

    harmful_train_states = collect_states_with_optional_ablation(
        model,
        tokenizer,
        harmful_train,
        target_layer=TARGET_LAYER,
        desc="exp18.train_harmful",
    )
    harmless_train_states = collect_states_with_optional_ablation(
        model,
        tokenizer,
        harmless_train,
        target_layer=TARGET_LAYER,
        desc="exp18.train_harmless",
    )
    harmful_train_ablated_states = collect_states_with_optional_ablation(
        model,
        tokenizer,
        harmful_train,
        direction=r_exec,
        abl_layers=ABL_LAYERS,
        target_layer=TARGET_LAYER,
        desc="exp18.train_harmful_ablated",
    )
    harmless_train_ablated_states = collect_states_with_optional_ablation(
        model,
        tokenizer,
        harmless_train,
        direction=r_exec,
        abl_layers=ABL_LAYERS,
        target_layer=TARGET_LAYER,
        desc="exp18.train_harmless_ablated",
    )

    r_detect_raw = mean_diff_direction(
        {TARGET_LAYER: harmful_train_ablated_states},
        {TARGET_LAYER: harmless_train_ablated_states},
    )[TARGET_LAYER].to(device)
    r_detect = remove_projection(
        {TARGET_LAYER: r_detect_raw},
        {TARGET_LAYER: r_exec},
    )[TARGET_LAYER].to(device)

    detect_stability = split_half_stability(
        {TARGET_LAYER: harmful_train_ablated_states},
        {TARGET_LAYER: harmless_train_ablated_states},
        k=20,
        seed=args.seed,
    )[TARGET_LAYER]
    exec_stability = split_half_stability(
        {TARGET_LAYER: harmful_train_states},
        {TARGET_LAYER: harmless_train_states},
        k=20,
        seed=args.seed,
    )[TARGET_LAYER]

    r_exec_cpu = r_exec.detach().float().cpu()
    r_detect_cpu = r_detect.detach().float().cpu()

    base_proj_exec = projection_values(
        {TARGET_LAYER: harmful_train_states},
        {TARGET_LAYER: r_exec_cpu},
    )[TARGET_LAYER].numpy()
    base_safe_proj_exec = projection_values(
        {TARGET_LAYER: harmless_train_states},
        {TARGET_LAYER: r_exec_cpu},
    )[TARGET_LAYER].numpy()
    base_proj_detect = projection_values(
        {TARGET_LAYER: harmful_train_states},
        {TARGET_LAYER: r_detect_cpu},
    )[TARGET_LAYER].numpy()
    base_safe_proj_detect = projection_values(
        {TARGET_LAYER: harmless_train_states},
        {TARGET_LAYER: r_detect_cpu},
    )[TARGET_LAYER].numpy()
    abl_proj_exec = projection_values(
        {TARGET_LAYER: harmful_train_ablated_states},
        {TARGET_LAYER: r_exec_cpu},
    )[TARGET_LAYER].numpy()
    abl_safe_proj_exec = projection_values(
        {TARGET_LAYER: harmless_train_ablated_states},
        {TARGET_LAYER: r_exec_cpu},
    )[TARGET_LAYER].numpy()
    abl_proj_detect = projection_values(
        {TARGET_LAYER: harmful_train_ablated_states},
        {TARGET_LAYER: r_detect_cpu},
    )[TARGET_LAYER].numpy()
    abl_safe_proj_detect = projection_values(
        {TARGET_LAYER: harmless_train_ablated_states},
        {TARGET_LAYER: r_detect_cpu},
    )[TARGET_LAYER].numpy()

    train_geometry = {
        "baseline_states": {
            "r_exec": summarize_direction_on_split(base_proj_exec, base_safe_proj_exec),
            "r_detect": summarize_direction_on_split(base_proj_detect, base_safe_proj_detect),
        },
        "exec_ablated_states": {
            "r_exec": summarize_direction_on_split(abl_proj_exec, abl_safe_proj_exec),
            "r_detect": summarize_direction_on_split(abl_proj_detect, abl_safe_proj_detect),
        },
    }

    # Part B: align prompt-time projections with saved Exp16 ShieldGemma results.
    harmful_rows = flatten_prompt_groups_with_topics(exp16_payload["prompt_groups"], "harmful")
    harmful_prompts = [row["prompt"] for row in harmful_rows]
    if args.n_eval_harmful > 0:
        harmful_rows = harmful_rows[: args.n_eval_harmful]
        harmful_prompts = harmful_prompts[: args.n_eval_harmful]

    baseline_prompt_states = collect_states_with_optional_ablation(
        model,
        tokenizer,
        harmful_prompts,
        target_layer=TARGET_LAYER,
        desc="exp18.exp16_harmful_baseline_states",
    )
    ablated_prompt_states = collect_states_with_optional_ablation(
        model,
        tokenizer,
        harmful_prompts,
        direction=r_exec,
        abl_layers=ABL_LAYERS,
        target_layer=TARGET_LAYER,
        desc="exp18.exp16_harmful_exec_ablated_states",
    )

    baseline_proj_exec_saved = projection_values(
        {TARGET_LAYER: baseline_prompt_states},
        {TARGET_LAYER: r_exec_cpu},
    )[TARGET_LAYER].numpy()
    baseline_proj_detect_saved = projection_values(
        {TARGET_LAYER: baseline_prompt_states},
        {TARGET_LAYER: r_detect_cpu},
    )[TARGET_LAYER].numpy()
    ablated_proj_exec_saved = projection_values(
        {TARGET_LAYER: ablated_prompt_states},
        {TARGET_LAYER: r_exec_cpu},
    )[TARGET_LAYER].numpy()
    ablated_proj_detect_saved = projection_values(
        {TARGET_LAYER: ablated_prompt_states},
        {TARGET_LAYER: r_detect_cpu},
    )[TARGET_LAYER].numpy()

    baseline_audit_items = exp16_payload["shield_audit"][BASELINE_SAFE_RESPONSE_GROUP]["items"][
        : len(harmful_rows)
    ]
    exec_audit_items = exp16_payload["shield_audit"][EXEC_ABLATED_RESPONSE_GROUP]["items"][
        : len(harmful_rows)
    ]
    baseline_span_presence = span_presence_by_response(exp16_payload, BASELINE_SAFE_RESPONSE_GROUP)
    exec_span_presence = span_presence_by_response(exp16_payload, EXEC_ABLATED_RESPONSE_GROUP)

    per_prompt_records = build_per_prompt_records(
        harmful_rows,
        baseline_proj_exec_saved,
        baseline_proj_detect_saved,
        ablated_proj_exec_saved,
        ablated_proj_detect_saved,
        baseline_audit_items,
        exec_audit_items,
        baseline_span_presence,
        exec_span_presence,
    )
    saved_prompt_metrics = summarize_saved_prompt_metrics(per_prompt_records)

    output = {
        "model": args.model,
        "seed": args.seed,
        "target_layer": TARGET_LAYER,
        "abl_layers": ABL_LAYERS,
        "n_train": args.n_train,
        "n_eval_harmful": len(harmful_rows),
        "exp16_input": args.exp16_input,
        "direction_summary": {
            "cosine_exec_detect_raw": cosine(r_exec, r_detect_raw),
            "cosine_exec_detect_orth": cosine(r_exec, r_detect),
            "r_exec_norm": float(r_exec.norm().item()),
            "r_detect_raw_norm": float(r_detect_raw.norm().item()),
            "r_detect_norm": float(r_detect.norm().item()),
            "r_detect_collapsed_after_orthogonalization": bool(r_detect.norm().item() < 1e-6),
            "r_exec_stability": {
                "mean": safe_float(exec_stability["mean"]),
                "std": safe_float(exec_stability["std"]),
            },
            "r_detect_stability": {
                "mean": safe_float(detect_stability["mean"]),
                "std": safe_float(detect_stability["std"]),
            },
        },
        "train_geometry": train_geometry,
        "saved_prompt_metrics": saved_prompt_metrics,
        "per_prompt_records": per_prompt_records,
    }
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[exp_18] saved={args.output}")


if __name__ == "__main__":
    main()
