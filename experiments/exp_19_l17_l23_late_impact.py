"""
experiments/exp_19_l17_l23_late_impact.py
=========================================
Probe how L17 and L23 ablations causally reshape:
  1. harmful-output risk under ShieldGemma,
  2. natural response composition (refusal / empathy / resource / unsafe spans),
  3. prompt-time late-layer sparse features and output-time late-layer sparse features.

This experiment keeps Gemma-3-1B-IT as the mechanism target. ShieldGemma is
only the external safety auditor.
"""

from __future__ import annotations

import argparse
import json
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
from data.topic_banks import load_topic_banks
from probes.ablate import _make_ablate_hook
from probes.direction_cache import extract_and_cache
from probes.extract import _build_prompt, mean_diff_direction, remove_projection
from probes.model_structure import get_transformer_layer
from probes.gemma_scope import (
    build_scope_release,
    encode_scope_features,
    preload_scope_saes,
    summarize_feature_activations,
    summarize_feature_contrast,
)
from probes.response_spans import (
    build_span_records,
    collect_segment_hidden_states,
    summarize_span_records,
)
from probes.shield_audit import (
    AuditItem,
    ShieldGemmaAuditor,
    pick_top_audit_candidates,
    summarize_audit_results,
)
from probes.stats import set_seed


TARGET_LAYER = 17
LATE_LAYERS = [18, 19, 20, 21, 22, 23, 24]
ABL_LAYERS = [17, 23]
KEY_SPAN_LABELS = [
    "refusal_clause",
    "risk_warning",
    "empathy_apology",
    "redirect_clause",
    "resource_redirect",
    "resource_list",
    "unsafe_instructions",
]
FAMILY_GROUP_SPECS = {
    "empathy_family": "harmful_baseline:empathy_apology",
    "refusal_family": "harmful_baseline:refusal_clause",
    "risk_family": "harmful_baseline:risk_warning",
    "resource_family": "harmful_baseline:resource_redirect",
    "unsafe_exec_family": "harmful_exec_only:unsafe_instructions",
}
PROMPT_CONTRASTS = [
    ("baseline", "l17_only"),
    ("baseline", "l23_only"),
    ("baseline", "l17_l23"),
    ("baseline", "dual_l17_l23"),
    ("l17_only", "l23_only"),
    ("l17_only", "l17_l23"),
    ("l23_only", "l17_l23"),
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


def flatten_prompt_rows(
    topic_payload: Dict[str, Dict[str, List[str]]],
    group_name: str,
) -> List[dict]:
    rows: List[dict] = []
    for topic in sorted(topic_payload.keys()):
        for prompt in topic_payload[topic][group_name]:
            rows.append({"topic": topic, "prompt": prompt})
    return rows


def deterministic_generate(model, **kwargs):
    return model.generate(
        **kwargs,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )


def build_condition_hooks(
    condition_name: str,
    *,
    r_exec: torch.Tensor,
    r_l23: torch.Tensor,
) -> list[tuple[int, object]]:
    if condition_name == "baseline":
        return []
    if condition_name == "l17_only":
        return [(17, _make_ablate_hook(r_exec))]
    if condition_name == "l23_only":
        return [(23, _make_ablate_hook(r_exec))]
    if condition_name == "l17_l23":
        return [
            (17, _make_ablate_hook(r_exec)),
            (23, _make_ablate_hook(r_exec)),
        ]
    if condition_name == "dual_l17_l23":
        return [
            (17, _make_ablate_hook(r_exec)),
            (23, _make_ablate_hook(r_l23)),
        ]
    raise ValueError(f"Unsupported condition: {condition_name}")


def generate_with_hook_specs(
    model,
    tokenizer,
    prompt: str,
    *,
    hook_specs: Sequence[tuple[int, object]],
    max_new_tokens: int,
) -> str:
    text = _build_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
        handles = []
        try:
            for layer_idx, hook_fn in hook_specs:
                handles.append(get_transformer_layer(model, layer_idx).register_forward_hook(hook_fn))
            with torch.no_grad():
                out = deterministic_generate(
                    model,
                **inputs,
                max_new_tokens=max_new_tokens,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    finally:
        for handle in handles:
            handle.remove()


def generate_condition_outputs(
    model,
    tokenizer,
    prompt_rows: Sequence[dict],
    *,
    condition_name: str,
    r_exec: torch.Tensor,
    r_l23: torch.Tensor,
    max_new_tokens: int,
) -> list[str]:
    hook_specs = build_condition_hooks(
        condition_name,
        r_exec=r_exec,
        r_l23=r_l23,
    )
    responses = []
    for row in tqdm(prompt_rows, desc=f"exp19.gen.{condition_name}"):
        responses.append(
            generate_with_hook_specs(
                model,
                tokenizer,
                row["prompt"],
                hook_specs=hook_specs,
                max_new_tokens=max_new_tokens,
            )
        )
    return responses


def run_shield_audit(
    prompt_rows: Sequence[dict],
    responses: Sequence[str],
    *,
    source_path: str,
    truncate_response: int,
    auditor: ShieldGemmaAuditor | None = None,
) -> dict:
    auditor = auditor or ShieldGemmaAuditor()
    items = [
        AuditItem(
            prompt=row["prompt"],
            response=response,
            source_file="exp19_l17_l23_late_impact",
            source_path=source_path,
            meta={"topic": row["topic"]},
        )
        for row, response in zip(prompt_rows, responses)
    ]
    results = auditor.audit_items(
        items,
        truncate_response=truncate_response,
        progress=f"exp19.shield.{source_path}",
    )
    topic_groups: dict[str, list] = defaultdict(list)
    for result in results:
        topic = str((result.meta or {}).get("topic", "unknown"))
        topic_groups[topic].append(result)
    return {
        "summary": summarize_audit_results(results),
        "summary_by_topic": {
            topic: summarize_audit_results(topic_results)
            for topic, topic_results in sorted(topic_groups.items())
        },
        "top_candidates": pick_top_audit_candidates(results),
        "items": [result.to_dict() for result in results],
    }


def collect_prompt_states_with_hooks(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    capture_layers: Sequence[int],
    hook_specs: Sequence[tuple[int, object]],
    desc: str,
) -> Dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    state_lists: dict[int, list[torch.Tensor]] = {layer: [] for layer in capture_layers}

    for prompt in tqdm(prompts, desc=desc):
        text = _build_prompt(tokenizer, prompt)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        state_holder: dict[int, torch.Tensor] = {}
        handles = []
        try:
            for layer_idx, hook_fn in hook_specs:
                handles.append(get_transformer_layer(model, layer_idx).register_forward_hook(hook_fn))

            for layer in capture_layers:
                def make_capture(target_layer: int):
                    def capture_hook(module, inputs, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        state_holder[target_layer] = hidden[0, -1, :].float().cpu()
                        return output

                    return capture_hook

                handles.append(
                    get_transformer_layer(model, layer).register_forward_hook(make_capture(layer))
                )

            with torch.no_grad():
                model(
                    input_ids=input_ids,
                    output_hidden_states=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        for layer in capture_layers:
            if layer not in state_holder:
                raise RuntimeError(f"Failed to capture prompt state for layer {layer}.")
            state_lists[layer].append(state_holder[layer])

    return {
        layer: torch.stack(vectors, dim=0)
        for layer, vectors in state_lists.items()
    }


def summarize_response_presence(records: Sequence[dict]) -> dict:
    response_count = len(records)
    span_count = Counter()
    response_presence = Counter()
    by_topic_response_presence: dict[str, Counter] = defaultdict(Counter)

    for record in records:
        labels = set()
        topic = str(record.get("topic", "unknown"))
        for span in record.get("spans", []):
            label = span["label"]
            span_count[label] += 1
            labels.add(label)
        for label in labels:
            response_presence[label] += 1
            by_topic_response_presence[topic][label] += 1

    return {
        "n_responses": response_count,
        "span_count": dict(span_count),
        "response_presence_count": dict(response_presence),
        "response_presence_rate": {
            label: float(count / max(response_count, 1))
            for label, count in sorted(response_presence.items())
        },
        "response_presence_by_topic": {
            topic: {
                label: {
                    "count": count,
                    "rate": float(count / max(sum(1 for record in records if record.get("topic") == topic), 1)),
                }
                for label, count in sorted(counter.items())
            }
            for topic, counter in sorted(by_topic_response_presence.items())
        },
    }


def merge_states_for_prefix(
    group_states: Dict[str, Dict[int, torch.Tensor]],
    *,
    prefix: str,
    layers: Sequence[int],
) -> Dict[int, torch.Tensor]:
    merged: Dict[int, torch.Tensor] = {}
    for layer in layers:
        chunks = [
            layer_map[layer]
            for group_key, layer_map in group_states.items()
            if group_key.startswith(prefix) and layer in layer_map
        ]
        if chunks:
            merged[layer] = torch.cat(chunks, dim=0)
    return merged


def load_feature_families(
    exp17_input: str | Path,
    *,
    layers: Sequence[int],
    top_k: int,
) -> dict[str, dict[int, list[int]]]:
    payload = json.loads(Path(exp17_input).read_text(encoding="utf-8"))
    families: dict[str, dict[int, list[int]]] = {}
    for family_name, group_key in FAMILY_GROUP_SPECS.items():
        per_layer: dict[int, list[int]] = {}
        layer_payload = payload.get("group_feature_summary", {}).get(group_key, {}).get("layers", {})
        for layer in layers:
            items = layer_payload.get(str(layer), {}).get("top_mean_features", [])
            features = [int(item["feature"]) for item in items[:top_k]]
            if features:
                per_layer[layer] = features
        families[family_name] = per_layer

    safe_response_family: dict[int, list[int]] = {}
    for layer in layers:
        merged = []
        for family_name in ["refusal_family", "risk_family", "empathy_family", "resource_family"]:
            merged.extend(families.get(family_name, {}).get(layer, []))
        if merged:
            safe_response_family[layer] = sorted(set(merged))
    families["safe_response_family"] = safe_response_family
    return families


def summarize_feature_families(
    feature_acts: torch.Tensor,
    *,
    layer: int,
    feature_families: dict[str, dict[int, list[int]]],
    active_threshold: float = 1e-6,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for family_name, layer_map in feature_families.items():
        features = layer_map.get(layer, [])
        if not features:
            continue
        valid = [feature for feature in features if feature < feature_acts.shape[1]]
        if not valid:
            continue
        sub = feature_acts[:, valid]
        summary[family_name] = {
            "n_features": len(valid),
            "features": valid,
            "mean_activation": float(sub.mean().item()),
            "mean_prompt_max": float(sub.max(dim=1).values.mean().item()),
            "fire_rate": float((sub > active_threshold).any(dim=1).float().mean().item()),
        }
    return summary


def feature_family_scores(
    feature_acts: torch.Tensor,
    *,
    layer: int,
    feature_families: dict[str, dict[int, list[int]]],
) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    for family_name, layer_map in feature_families.items():
        features = layer_map.get(layer, [])
        valid = [feature for feature in features if feature < feature_acts.shape[1]]
        if not valid:
            continue
        sub = feature_acts[:, valid]
        scores[family_name] = sub.mean(dim=1).float().cpu().numpy()
    return scores


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(value):
        return None
    return value


def summarize_prompt_vector_family_correlations(
    condition_prompt_states: Dict[str, Dict[int, torch.Tensor]],
    *,
    z_exec_by_condition: Dict[str, np.ndarray],
    z_detect_by_condition: Dict[str, np.ndarray],
    scope_saes: Dict[int, object],
    feature_families: dict[str, dict[int, list[int]]],
    batch_size: int,
) -> tuple[dict, dict]:
    family_summary: dict[str, dict[str, dict[str, object]]] = {}
    feature_cache: dict[str, dict[int, torch.Tensor]] = {}

    for condition_name, layer_map in condition_prompt_states.items():
        family_summary[condition_name] = {}
        feature_cache[condition_name] = {}
        for layer, states in layer_map.items():
            if layer not in scope_saes:
                continue
            feature_acts = encode_scope_features(
                scope_saes[layer],
                states,
                batch_size=batch_size,
                desc=None,
            )
            feature_cache[condition_name][layer] = feature_acts
            family_summary[condition_name][str(layer)] = {
                "top_features": summarize_feature_activations(feature_acts, top_k=20),
                "family_activation": summarize_feature_families(
                    feature_acts,
                    layer=layer,
                    feature_families=feature_families,
                ),
                "vector_family_correlations": {
                    family_name: {
                        "corr_z_exec": safe_corr(z_exec_by_condition[condition_name], scores),
                        "corr_z_detect_candidate": safe_corr(z_detect_by_condition[condition_name], scores),
                    }
                    for family_name, scores in feature_family_scores(
                        feature_acts,
                        layer=layer,
                        feature_families=feature_families,
                    ).items()
                },
            }

    contrasts: dict[str, dict[str, object]] = {}
    for cond_a, cond_b in PROMPT_CONTRASTS:
        if cond_a not in feature_cache or cond_b not in feature_cache:
            continue
        key = f"{cond_a}__vs__{cond_b}"
        contrasts[key] = {}
        for layer in LATE_LAYERS:
            if layer not in feature_cache[cond_a] or layer not in feature_cache[cond_b]:
                continue
            contrasts[key][str(layer)] = summarize_feature_contrast(
                feature_cache[cond_a][layer],
                feature_cache[cond_b][layer],
                top_k=20,
            )
    return family_summary, contrasts


def summarize_output_feature_families(
    group_states: Dict[str, Dict[int, torch.Tensor]],
    *,
    scope_saes: Dict[int, object],
    feature_families: dict[str, dict[int, list[int]]],
    batch_size: int,
    min_group_size: int,
) -> tuple[dict, dict]:
    merged_by_condition: dict[str, Dict[int, torch.Tensor]] = {}
    for condition_name in ["baseline", "l17_only", "l23_only", "l17_l23", "dual_l17_l23"]:
        merged = merge_states_for_prefix(
            group_states,
            prefix=f"{condition_name}:",
            layers=LATE_LAYERS,
        )
        if merged:
            merged_by_condition[condition_name] = merged

    summary: dict[str, dict[str, object]] = {}
    feature_cache: dict[str, dict[int, torch.Tensor]] = {}
    for condition_name, layer_map in merged_by_condition.items():
        summary[condition_name] = {}
        feature_cache[condition_name] = {}
        for layer, states in layer_map.items():
            if states.shape[0] < min_group_size:
                continue
            feature_acts = encode_scope_features(
                scope_saes[layer],
                states,
                batch_size=batch_size,
                desc=None,
            )
            feature_cache[condition_name][layer] = feature_acts
            summary[condition_name][str(layer)] = {
                "n_spans": int(states.shape[0]),
                "top_features": summarize_feature_activations(feature_acts, top_k=20),
                "family_activation": summarize_feature_families(
                    feature_acts,
                    layer=layer,
                    feature_families=feature_families,
                ),
            }

    contrasts: dict[str, dict[str, object]] = {}
    for cond_a, cond_b in PROMPT_CONTRASTS:
        if cond_a not in feature_cache or cond_b not in feature_cache:
            continue
        key = f"{cond_a}__vs__{cond_b}"
        contrasts[key] = {}
        for layer in LATE_LAYERS:
            if layer not in feature_cache[cond_a] or layer not in feature_cache[cond_b]:
                continue
            contrasts[key][str(layer)] = summarize_feature_contrast(
                feature_cache[cond_a][layer],
                feature_cache[cond_b][layer],
                top_k=20,
            )
    return summary, contrasts


def build_condition_records(
    prompt_rows: Sequence[dict],
    responses: Sequence[str],
    condition_name: str,
) -> list[dict]:
    records = build_span_records(
        [row["prompt"] for row in prompt_rows],
        list(responses),
        condition_name,
    )
    for record, row in zip(records, prompt_rows):
        record["topic"] = row["topic"]
    return records


def collect_direction_candidate(
    model,
    tokenizer,
    *,
    r_exec: torch.Tensor,
    n_train: int,
    seed: int,
) -> torch.Tensor:
    harmful_train, harmless_train = load_default_datasets(
        n_harmful=n_train,
        n_harmless=n_train,
        split="train",
        seed=seed,
    )
    harmful_states = collect_prompt_states_with_hooks(
        model,
        tokenizer,
        harmful_train,
        capture_layers=[TARGET_LAYER],
        hook_specs=build_condition_hooks("l17_l23", r_exec=r_exec, r_l23=r_exec),
        desc="exp19.train_harmful_ablated",
    )[TARGET_LAYER]
    harmless_states = collect_prompt_states_with_hooks(
        model,
        tokenizer,
        harmless_train,
        capture_layers=[TARGET_LAYER],
        hook_specs=build_condition_hooks("l17_l23", r_exec=r_exec, r_l23=r_exec),
        desc="exp19.train_harmless_ablated",
    )[TARGET_LAYER]
    raw = mean_diff_direction(
        {TARGET_LAYER: harmful_states},
        {TARGET_LAYER: harmless_states},
    )[TARGET_LAYER].to(r_exec.device)
    return remove_projection(
        {TARGET_LAYER: raw},
        {TARGET_LAYER: r_exec},
    )[TARGET_LAYER].to(r_exec.device)


def projection_values_1d(states: torch.Tensor, direction: torch.Tensor) -> np.ndarray:
    vec = direction.detach().float().cpu()
    return (states.float().cpu() @ vec).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="all", choices=["train", "dev", "test", "all"])
    parser.add_argument("--n_per_group", type=int, default=999)
    parser.add_argument("--n_train_exec", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=180)
    parser.add_argument("--shield_truncate", type=int, default=500)
    parser.add_argument("--skip_shield_audit", action="store_true")
    parser.add_argument("--scope_site", default="res", choices=["res", "att", "mlp", "transcoder"])
    parser.add_argument("--scope_release", default=None)
    parser.add_argument("--scope_width", default="16k")
    parser.add_argument("--scope_l0", default="small")
    parser.add_argument("--scope_device", default="auto")
    parser.add_argument("--scope_dtype", default="float32")
    parser.add_argument("--scope_top_k_family", type=int, default=3)
    parser.add_argument("--scope_batch_size", type=int, default=128)
    parser.add_argument("--exp17_input", default="results/exp17_gemma_scope_feature_probe_full.json")
    parser.add_argument("--min_group_size", type=int, default=4)
    parser.add_argument("--output", default="results/exp19_l17_l23_late_impact.json")
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
    r_l23 = extract_and_cache(
        model,
        tokenizer,
        args.model,
        layers=[23],
        n_train=args.n_train_exec,
        seed=args.seed,
    )[23].to(device)
    r_detect_candidate = collect_direction_candidate(
        model,
        tokenizer,
        r_exec=r_exec,
        n_train=args.n_train_exec,
        seed=args.seed,
    )

    topic_payload = load_topic_banks(
        split=args.split,
        seed=args.seed,
        n_per_group=args.n_per_group,
    )
    prompt_rows = flatten_prompt_rows(topic_payload, "harmful")
    prompts = [row["prompt"] for row in prompt_rows]

    condition_outputs: dict[str, list[str]] = {}
    for condition_name in ["baseline", "l17_only", "l23_only", "l17_l23", "dual_l17_l23"]:
        condition_outputs[condition_name] = generate_condition_outputs(
            model,
            tokenizer,
            prompt_rows,
            condition_name=condition_name,
            r_exec=r_exec,
            r_l23=r_l23,
            max_new_tokens=args.max_new_tokens,
        )

    shield_audit: dict[str, object] = {}
    if not args.skip_shield_audit:
        for condition_name, responses in condition_outputs.items():
            shield_audit[condition_name] = run_shield_audit(
                prompt_rows,
                responses,
                source_path=condition_name,
                truncate_response=args.shield_truncate,
            )

    condition_records: dict[str, list[dict]] = {}
    all_records: list[dict] = []
    for condition_name, responses in condition_outputs.items():
        records = build_condition_records(prompt_rows, responses, condition_name)
        condition_records[condition_name] = records
        all_records.extend(records)

    span_summary = summarize_span_records(all_records)
    response_presence = {
        condition_name: summarize_response_presence(records)
        for condition_name, records in condition_records.items()
    }

    prompt_capture_layers = [TARGET_LAYER] + LATE_LAYERS
    condition_prompt_states: dict[str, Dict[int, torch.Tensor]] = {}
    z_exec_by_condition: dict[str, np.ndarray] = {}
    z_detect_by_condition: dict[str, np.ndarray] = {}
    for condition_name in ["baseline", "l17_only", "l23_only", "l17_l23", "dual_l17_l23"]:
        states = collect_prompt_states_with_hooks(
            model,
            tokenizer,
            prompts,
            capture_layers=prompt_capture_layers,
            hook_specs=build_condition_hooks(
                condition_name,
                r_exec=r_exec,
                r_l23=r_l23,
            ),
            desc=f"exp19.prompt_states.{condition_name}",
        )
        condition_prompt_states[condition_name] = states
        z_exec_by_condition[condition_name] = projection_values_1d(states[TARGET_LAYER], r_exec)
        z_detect_by_condition[condition_name] = projection_values_1d(states[TARGET_LAYER], r_detect_candidate)

    group_states, group_metadata = collect_segment_hidden_states(
        model,
        tokenizer,
        all_records,
        layers=LATE_LAYERS,
        desc="exp19.span_states",
    )

    resolved_scope_device = args.scope_device
    if resolved_scope_device == "auto":
        resolved_scope_device = "cuda" if torch.cuda.is_available() else "cpu"
    release = args.scope_release or build_scope_release(
        args.model,
        site=args.scope_site,
        all_layers=True,
    )
    scope_saes, scope_infos = preload_scope_saes(
        LATE_LAYERS,
        release=release,
        width=args.scope_width,
        l0=args.scope_l0,
        device=resolved_scope_device,
        dtype=args.scope_dtype,
        force_download=False,
    )
    feature_families = load_feature_families(
        args.exp17_input,
        layers=LATE_LAYERS,
        top_k=args.scope_top_k_family,
    )

    prompt_feature_summary, prompt_feature_contrasts = summarize_prompt_vector_family_correlations(
        condition_prompt_states,
        z_exec_by_condition=z_exec_by_condition,
        z_detect_by_condition=z_detect_by_condition,
        scope_saes=scope_saes,
        feature_families=feature_families,
        batch_size=args.scope_batch_size,
    )
    output_feature_summary, output_feature_contrasts = summarize_output_feature_families(
        group_states,
        scope_saes=scope_saes,
        feature_families=feature_families,
        batch_size=args.scope_batch_size,
        min_group_size=args.min_group_size,
    )

    output = {
        "model": args.model,
        "seed": args.seed,
        "split": args.split,
        "n_prompts": len(prompt_rows),
        "prompt_groups": topic_payload,
        "conditions": {
            condition_name: {
                "n": len(responses),
                "responses": responses,
            }
            for condition_name, responses in condition_outputs.items()
        },
        "direction_summary": {
            "r_exec_norm": float(r_exec.norm().item()),
            "r_l23_norm": float(r_l23.norm().item()),
            "cosine_r_exec_r_l23": float(torch.dot(r_exec, r_l23).item()),
            "r_detect_candidate_norm": float(r_detect_candidate.norm().item()),
            "cosine_r_exec_r_detect_candidate": float(torch.dot(r_exec, r_detect_candidate).item()),
        },
        "shield_audit": shield_audit,
        "span_summary": span_summary,
        "response_presence": response_presence,
        "feature_families": {
            family_name: {str(layer): features for layer, features in layer_map.items()}
            for family_name, layer_map in feature_families.items()
        },
        "scope_release": release,
        "scope_width": args.scope_width,
        "scope_l0": args.scope_l0,
        "scope_device": resolved_scope_device,
        "scope_infos": {
            str(layer): info.to_dict()
            for layer, info in scope_infos.items()
        },
        "prompt_state_summary": {
            "z_exec_by_condition": {
                condition_name: {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
                for condition_name, values in z_exec_by_condition.items()
            },
            "z_detect_candidate_by_condition": {
                condition_name: {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
                for condition_name, values in z_detect_by_condition.items()
            },
            "late_feature_summary": prompt_feature_summary,
            "late_feature_contrasts": prompt_feature_contrasts,
        },
        "output_state_summary": {
            "late_feature_summary": output_feature_summary,
            "late_feature_contrasts": output_feature_contrasts,
        },
    }
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[exp_19] saved={args.output}")


if __name__ == "__main__":
    main()
