"""
experiments/exp_26_vector_interaction_map.py
===========================================
Map second-order interactions between:

1. the refusal execution gate (`r_exec`) at L17/L23,
2. a topic-specific late safe-response vector at a late layer.

This experiment is intentionally mechanism-first:
  - local harmful prompts are used only as internal stimuli,
  - dangerous continuations are never written to disk,
  - outputs are summarized as risk scores, span-label counts, and interaction terms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.topic_banks import load_topic_banks  # noqa: E402
from experiments.exp_19_l17_l23_late_impact import (  # noqa: E402
    LATE_LAYERS,
    TARGET_LAYER,
    build_scope_release,
    feature_family_scores,
    load_feature_families,
    load_model,
    projection_values_1d,
)
from probes.ablate import _make_ablate_hook, _make_addition_hook  # noqa: E402
from probes.direction_cache import extract_and_cache  # noqa: E402
from probes.extract import _build_prompt, collect_hidden_states, mean_diff_direction, remove_projection  # noqa: E402
from probes.gemma_scope import build_scope_sae_id, encode_scope_features, load_scope_sae  # noqa: E402
from probes.response_spans import build_span_records  # noqa: E402
from probes.shield_audit import (  # noqa: E402
    DIRECT_DANGER_FEWSHOT_KEY,
    AuditItem,
    HarmType,
    ShieldGemmaAuditor,
    default_prompt_specs,
    direct_danger_fewshot_spec,
)
from probes.stats import set_seed  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
POC_ROOT = ROOT / "poc"
ALL_LAYERS = list(range(26))
DEFAULT_TOPICS = ["self_harm", "explosives", "fraud", "hate_or_abuse"]
DEFAULT_CONDITIONS = ["baseline", "exec_off", "late_safe_on", "exec_off_plus_late_safe_on"]
LATE_SAFE_LAYER = 23


def resolve_runtime_path(raw_path: str, *, for_output: bool = False) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if for_output:
        return Path.cwd() / path
    candidates = [
        Path.cwd() / path,
        ROOT / path,
        POC_ROOT / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def parse_name_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def deterministic_generate(model, **kwargs):
    return model.generate(
        **kwargs,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )


def collect_topic_late_safe_vectors(
    model,
    tokenizer,
    *,
    topic_payload_train: dict[str, dict[str, list[str]]],
    r_exec: torch.Tensor,
) -> dict[str, torch.Tensor]:
    topic_vectors: dict[str, torch.Tensor] = {}
    for topic, groups in sorted(topic_payload_train.items()):
        harmful_states = collect_hidden_states(
            model,
            tokenizer,
            groups["harmful"],
            layers=[LATE_SAFE_LAYER],
            desc=f"exp26.{topic}.harm.train",
        )[LATE_SAFE_LAYER]
        supportive_states = collect_hidden_states(
            model,
            tokenizer,
            groups["supportive"],
            layers=[LATE_SAFE_LAYER],
            desc=f"exp26.{topic}.support.train",
        )[LATE_SAFE_LAYER]
        direction = mean_diff_direction(
            {LATE_SAFE_LAYER: supportive_states},
            {LATE_SAFE_LAYER: harmful_states},
        )[LATE_SAFE_LAYER]
        direction = remove_projection(
            {LATE_SAFE_LAYER: direction},
            {LATE_SAFE_LAYER: r_exec},
        )[LATE_SAFE_LAYER]
        topic_vectors[topic] = direction
    return topic_vectors


def build_condition_hooks(
    *,
    condition_name: str,
    r_exec: torch.Tensor,
    late_safe_vec: torch.Tensor,
    late_safe_alpha: float,
) -> list[tuple[int, object]]:
    if condition_name == "baseline":
        return []
    if condition_name == "exec_off":
        return [
            (17, _make_ablate_hook(r_exec)),
            (23, _make_ablate_hook(r_exec)),
        ]
    if condition_name == "late_safe_on":
        return [
            (LATE_SAFE_LAYER, _make_addition_hook(late_safe_vec, alpha=late_safe_alpha)),
        ]
    if condition_name == "exec_off_plus_late_safe_on":
        return [
            (17, _make_ablate_hook(r_exec)),
            (23, _make_ablate_hook(r_exec)),
            (LATE_SAFE_LAYER, _make_addition_hook(late_safe_vec, alpha=late_safe_alpha)),
        ]
    raise ValueError(f"Unsupported condition: {condition_name}")


def capture_prompt_end_states(
    model,
    tokenizer,
    *,
    prompts: Sequence[str],
    hook_specs: Sequence[tuple[int, object]],
    capture_layers: Sequence[int],
    desc: str,
) -> dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    state_lists: dict[int, list[torch.Tensor]] = {layer: [] for layer in capture_layers}
    for prompt in tqdm(prompts, desc=desc):
        text = _build_prompt(tokenizer, prompt)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        holder: dict[int, torch.Tensor] = {}
        handles = []
        try:
            for layer_idx, hook_fn in hook_specs:
                handles.append(model.model.layers[layer_idx].register_forward_hook(hook_fn))
            for layer in capture_layers:
                def make_capture(target_layer: int):
                    def capture_hook(module, inputs, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        holder[target_layer] = hidden[0, -1, :].float().cpu()
                        return output

                    return capture_hook

                handles.append(model.model.layers[layer].register_forward_hook(make_capture(layer)))
            with torch.no_grad():
                model(input_ids=input_ids, output_hidden_states=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        for layer in capture_layers:
            state_lists[layer].append(holder[layer])
    return {
        layer: torch.stack(vectors, dim=0)
        for layer, vectors in state_lists.items()
    }


def generate_responses(
    model,
    tokenizer,
    *,
    prompts: Sequence[str],
    hook_specs: Sequence[tuple[int, object]],
    max_new_tokens: int,
    desc: str,
) -> list[str]:
    responses: list[str] = []
    for prompt in tqdm(prompts, desc=desc):
        text = _build_prompt(tokenizer, prompt)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        handles = []
        try:
            for layer_idx, hook_fn in hook_specs:
                handles.append(model.model.layers[layer_idx].register_forward_hook(hook_fn))
            with torch.no_grad():
                out = deterministic_generate(model, **inputs, max_new_tokens=max_new_tokens)
            new_tokens = out[0, inputs["input_ids"].shape[1] :]
            responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        finally:
            for handle in handles:
                handle.remove()
    return responses


def summarize_span_labels(span_records: Sequence[dict]) -> dict[str, Any]:
    counts = Counter()
    first_labels = Counter()
    for record in span_records:
        spans = record.get("spans", [])
        if spans:
            first_labels[spans[0]["label"]] += 1
        for span in spans:
            counts[span["label"]] += 1
    return {
        "span_counts": dict(counts),
        "first_label_counts": dict(first_labels),
    }


def summarize_scores(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row["audit_scores"][key]) for row in rows]
    return {
        "mean": round(sum(values) / max(len(values), 1), 4),
        "max": round(max(values, default=0.0), 4),
        "over_0_5": sum(value >= 0.5 for value in values),
        "over_0_8": sum(value >= 0.8 for value in values),
    }


def summarize_prompt_geometry(
    prompt_states: dict[int, torch.Tensor],
    *,
    r_exec: torch.Tensor,
    late_safe_vec: torch.Tensor,
    feature_families: dict[str, dict[int, list[int]]] | None,
    scope_release: str | None,
    scope_width: str,
    scope_l0: str,
    scope_device: str,
    scope_dtype: str,
    scope_batch_size: int,
) -> dict[str, Any]:
    target_states = prompt_states[TARGET_LAYER]
    out = {
        "z_exec_mean": float(projection_values_1d(target_states, r_exec).mean()),
        "z_late_safe_mean": float(projection_values_1d(prompt_states[LATE_SAFE_LAYER], late_safe_vec).mean()),
        "late_family_means": {},
    }
    if feature_families is None or scope_release is None:
        return out
    layer_states = prompt_states[LATE_SAFE_LAYER]
    sae, _ = load_scope_sae(
        release=scope_release,
        sae_id=build_scope_sae_id(LATE_SAFE_LAYER, width=scope_width, l0=scope_l0),
        device=scope_device,
        dtype=scope_dtype,
    )
    feature_acts = encode_scope_features(
        sae,
        layer_states,
        batch_size=scope_batch_size,
    )
    layer_scores = feature_family_scores(
        feature_acts,
        layer=LATE_SAFE_LAYER,
        feature_families=feature_families,
    )
    out["late_family_means"] = {
        family_name: float(values.mean())
        for family_name, values in layer_scores.items()
    }
    try:
        sae.to("cpu")
    except Exception:
        pass
    del feature_acts
    del sae
    return out


def summarize_condition_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "dangerous": summarize_scores(rows, "dangerous"),
        DIRECT_DANGER_FEWSHOT_KEY: summarize_scores(rows, DIRECT_DANGER_FEWSHOT_KEY),
    }


def extract_mean(summary: dict[str, Any], key: str) -> float:
    return float(summary[key]["mean"])


def interaction_terms(condition_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = condition_summaries["baseline"]
    exec_off = condition_summaries["exec_off"]
    late_on = condition_summaries["late_safe_on"]
    combo = condition_summaries["exec_off_plus_late_safe_on"]
    out: dict[str, Any] = {}
    for key in ["dangerous", DIRECT_DANGER_FEWSHOT_KEY]:
        b = extract_mean(base, key)
        e = extract_mean(exec_off, key)
        l = extract_mean(late_on, key)
        c = extract_mean(combo, key)
        out[key] = {
            "main_effect_exec_off": round(e - b, 4),
            "main_effect_late_safe_on": round(l - b, 4),
            "late_safe_rescue_under_exec_off": round(c - e, 4),
            "interaction": round(c - e - l + b, 4),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topics", default=",".join(DEFAULT_TOPICS))
    parser.add_argument("--n_train_per_group", type=int, default=6)
    parser.add_argument("--n_eval_per_group", type=int, default=4)
    parser.add_argument("--late_safe_alpha", type=float, default=4.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--shield_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--scope_site", default="res", choices=["res", "att", "mlp", "transcoder"])
    parser.add_argument("--scope_release", default=None)
    parser.add_argument("--scope_width", default="16k")
    parser.add_argument("--scope_l0", default="small")
    parser.add_argument("--scope_device", default="auto")
    parser.add_argument("--scope_dtype", default="float32")
    parser.add_argument("--scope_batch_size", type=int, default=128)
    parser.add_argument("--scope_top_k_family", type=int, default=3)
    parser.add_argument("--exp17_input", default="results/exp17_gemma_scope_feature_probe_full.json")
    parser.add_argument("--n_train_exec", type=int, default=100)
    parser.add_argument("--output", default="results/exp26_vector_interaction_map.json")
    args = parser.parse_args()

    set_seed(args.seed)
    topics = parse_name_list(args.topics)
    model, tokenizer = load_model(args.model, args.hf_token)

    r_exec = extract_and_cache(
        model,
        tokenizer,
        layers=[TARGET_LAYER],
        n_train=args.n_train_exec,
        seed=args.seed,
        model_name=args.model,
    )[TARGET_LAYER]

    train_payload = load_topic_banks(
        split="train",
        seed=args.seed,
        topics=topics,
        n_per_group=args.n_train_per_group,
    )
    eval_payload = load_topic_banks(
        split="test",
        seed=args.seed,
        topics=topics,
        n_per_group=args.n_eval_per_group,
    )

    late_safe_vecs = collect_topic_late_safe_vectors(
        model,
        tokenizer,
        topic_payload_train=train_payload,
        r_exec=r_exec,
    )

    resolved_shield_device = args.shield_device
    if resolved_shield_device == "auto":
        resolved_shield_device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_scope_device = args.scope_device
    if resolved_scope_device == "auto":
        resolved_scope_device = "cuda" if torch.cuda.is_available() else "cpu"

    shield_auditor = ShieldGemmaAuditor(
        prompt_specs=[
            *default_prompt_specs([HarmType.DANGEROUS]),
            direct_danger_fewshot_spec(),
        ],
        device=resolved_shield_device,
    )

    feature_families = load_feature_families(
        resolve_runtime_path(args.exp17_input),
        layers=LATE_LAYERS,
        top_k=args.scope_top_k_family,
    )
    scope_release = args.scope_release or build_scope_release(args.model, site=args.scope_site)

    results_by_topic: dict[str, Any] = {}
    aggregate_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for topic in topics:
        harmful_prompts = eval_payload[topic]["harmful"]
        late_safe_vec = late_safe_vecs[topic]
        topic_condition_rows: dict[str, list[dict[str, Any]]] = {}
        topic_prompt_geometry: dict[str, Any] = {}
        topic_span_summaries: dict[str, Any] = {}

        for condition_name in DEFAULT_CONDITIONS:
            hook_specs = build_condition_hooks(
                condition_name=condition_name,
                r_exec=r_exec,
                late_safe_vec=late_safe_vec,
                late_safe_alpha=args.late_safe_alpha,
            )
            prompt_states = capture_prompt_end_states(
                model,
                tokenizer,
                prompts=harmful_prompts,
                hook_specs=hook_specs,
                capture_layers=[TARGET_LAYER, LATE_SAFE_LAYER],
                desc=f"exp26.prompt.{topic}.{condition_name}",
            )
            topic_prompt_geometry[condition_name] = summarize_prompt_geometry(
                prompt_states,
                r_exec=r_exec,
                late_safe_vec=late_safe_vec,
                feature_families=feature_families,
                scope_release=scope_release,
                scope_width=args.scope_width,
                scope_l0=args.scope_l0,
                scope_device=resolved_scope_device,
                scope_dtype=args.scope_dtype,
                scope_batch_size=args.scope_batch_size,
            )

            responses = generate_responses(
                model,
                tokenizer,
                prompts=harmful_prompts,
                hook_specs=hook_specs,
                max_new_tokens=args.max_new_tokens,
                desc=f"exp26.gen.{topic}.{condition_name}",
            )
            span_records = build_span_records(
                harmful_prompts,
                responses,
                source_group="harmful_exec_only",
            )
            topic_span_summaries[condition_name] = summarize_span_labels(span_records)

            audit_items = [
                AuditItem(
                    prompt=prompt,
                    response=response,
                    source_file="exp26_vector_interaction_map",
                    source_path=f"{topic}.{condition_name}[{idx}]",
                )
                for idx, (prompt, response) in enumerate(zip(harmful_prompts, responses))
            ]
            audit_results = shield_auditor.audit_items(
                audit_items,
                truncate_response=500,
                progress=f"exp26.audit.{topic}.{condition_name}",
            )
            rows: list[dict[str, Any]] = []
            for prompt, span_record, audit_result in zip(harmful_prompts, span_records, audit_results):
                row = {
                    "topic": topic,
                    "condition_name": condition_name,
                    "prompt": prompt,
                    "span_labels": [span["label"] for span in span_record.get("spans", [])],
                    "audit_scores": {
                        name: float(score.yes_prob)
                        for name, score in audit_result.scores.items()
                    },
                }
                rows.append(row)
            topic_condition_rows[condition_name] = rows
            aggregate_rows[condition_name].extend(rows)

        condition_summaries = {
            condition_name: summarize_condition_rows(rows)
            for condition_name, rows in topic_condition_rows.items()
        }
        results_by_topic[topic] = {
            "late_safe_alpha": args.late_safe_alpha,
            "late_safe_vector_layer": LATE_SAFE_LAYER,
            "condition_prompt_geometry": topic_prompt_geometry,
            "condition_span_summaries": topic_span_summaries,
            "condition_summaries": condition_summaries,
            "interaction_terms": interaction_terms(condition_summaries),
        }

    shield_auditor.unload()

    aggregate_condition_summaries = {
        condition_name: summarize_condition_rows(rows)
        for condition_name, rows in aggregate_rows.items()
    }
    output = {
        "model": args.model,
        "seed": args.seed,
        "topics": topics,
        "late_safe_alpha": args.late_safe_alpha,
        "conditions": DEFAULT_CONDITIONS,
        "note": "No generated responses are saved; only risk scores and label summaries are retained.",
        "aggregate_condition_summaries": aggregate_condition_summaries,
        "aggregate_interaction_terms": interaction_terms(aggregate_condition_summaries),
        "per_topic": results_by_topic,
    }

    output_path = resolve_runtime_path(args.output, for_output=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["aggregate_interaction_terms"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
