"""
experiments/exp_28_detect_family_causal.py
==========================================
Causally probe whether the detect-side direction reshapes downstream safety families.

This experiment keeps the same non-leaking discipline as Exp27:
  - risky prompts are used only as internal stimuli,
  - raw continuations are not written to disk,
  - outputs are summarized into labels, risk scores, and prompt-time geometry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.topic_banks import load_topic_banks  # noqa: E402
from experiments.exp_19_l17_l23_late_impact import (  # noqa: E402
    TARGET_LAYER,
    collect_direction_candidate,
    feature_family_scores,
    load_feature_families,
    load_model,
    projection_values_1d,
)
from probes.ablate import _make_ablate_hook, _make_addition_hook  # noqa: E402
from probes.direction_cache import extract_and_cache  # noqa: E402
from probes.gemma_scope import build_scope_release, build_scope_sae_id, encode_scope_features, load_scope_sae  # noqa: E402
from probes.extract import _build_prompt  # noqa: E402
from probes.review_labels import label_response, summarize_label_records  # noqa: E402
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
LATE_FAMILY_LAYER = 23
DEFAULT_TOPICS = ["self_harm", "explosives", "fraud", "hate_or_abuse"]
DEFAULT_GROUPS = ["harmful", "supportive", "safe_info"]
DEFAULT_CONDITIONS = [
    "baseline",
    "detect_off",
    "detect_add",
    "exec_off",
    "exec_off_plus_detect_add",
]


def resolve_runtime_path(raw_path: str, *, for_output: bool = False) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if for_output:
        return Path.cwd() / path
    candidates = [Path.cwd() / path, ROOT / path, POC_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def parse_name_list(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("Expected at least one name.")
    return names


def deterministic_generate(model, **kwargs):
    return model.generate(
        **kwargs,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )


def build_eval_rows(
    topic_payload: dict[str, dict[str, list[str]]],
    *,
    groups: Sequence[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for topic in sorted(topic_payload.keys()):
        for group_name in groups:
            for prompt in topic_payload[topic][group_name]:
                rows.append(
                    {
                        "topic": topic,
                        "group": group_name,
                        "source_group": f"{topic}:{group_name}",
                        "prompt": prompt,
                    }
                )
    return rows


def build_condition_hooks(
    *,
    condition_name: str,
    r_exec: torch.Tensor,
    r_detect: torch.Tensor,
    detect_alpha: float,
) -> list[tuple[int, object]]:
    if condition_name == "baseline":
        return []
    if condition_name == "detect_off":
        return [(TARGET_LAYER, _make_ablate_hook(r_detect))]
    if condition_name == "detect_add":
        return [(TARGET_LAYER, _make_addition_hook(r_detect, alpha=detect_alpha))]
    if condition_name == "exec_off":
        return [
            (17, _make_ablate_hook(r_exec)),
            (23, _make_ablate_hook(r_exec)),
        ]
    if condition_name == "exec_off_plus_detect_add":
        return [
            (17, _make_ablate_hook(r_exec)),
            (23, _make_ablate_hook(r_exec)),
            (TARGET_LAYER, _make_addition_hook(r_detect, alpha=detect_alpha)),
        ]
    raise ValueError(f"Unsupported condition: {condition_name}")


def run_condition_for_rows(
    model,
    tokenizer,
    *,
    rows: Sequence[dict[str, str]],
    condition_name: str,
    r_exec: torch.Tensor,
    r_detect: torch.Tensor,
    detect_alpha: float,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[int, torch.Tensor]]:
    device = next(model.parameters()).device
    prompt_state_lists: dict[int, list[torch.Tensor]] = {
        TARGET_LAYER: [],
        LATE_FAMILY_LAYER: [],
    }
    row_results: list[dict[str, Any]] = []

    for row in tqdm(rows, desc=f"exp28.{condition_name}"):
        hook_specs = build_condition_hooks(
            condition_name=condition_name,
            r_exec=r_exec,
            r_detect=r_detect,
            detect_alpha=detect_alpha,
        )
        holder: dict[int, torch.Tensor] = {}
        text = _build_prompt(tokenizer, row["prompt"])
        inputs = tokenizer(text, return_tensors="pt").to(device)
        handles = []
        try:
            for layer_idx, hook_fn in hook_specs:
                handles.append(model.model.layers[layer_idx].register_forward_hook(hook_fn))

            for layer in [TARGET_LAYER, LATE_FAMILY_LAYER]:
                def make_capture(target_layer: int):
                    def capture_hook(module, inputs, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        holder[target_layer] = hidden[0, -1, :].float().cpu()
                        return output

                    return capture_hook

                handles.append(
                    model.model.layers[layer].register_forward_hook(make_capture(layer))
                )

            with torch.no_grad():
                out = deterministic_generate(
                    model,
                    **inputs,
                    max_new_tokens=max_new_tokens,
                )
            new_tokens = out[0, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        finally:
            for handle in handles:
                handle.remove()

        prompt_state_lists[TARGET_LAYER].append(holder[TARGET_LAYER])
        prompt_state_lists[LATE_FAMILY_LAYER].append(holder[LATE_FAMILY_LAYER])
        row_results.append(
            {
                "topic": row["topic"],
                "group": row["group"],
                "source_group": row["source_group"],
                "prompt": row["prompt"],
                "response": response,
                "label": label_response(response, prompt=row["prompt"]),
            }
        )

    return row_results, {
        layer: torch.stack(vectors, dim=0)
        for layer, vectors in prompt_state_lists.items()
    }


def summarize_prompt_geometry(
    prompt_states: dict[int, torch.Tensor],
    *,
    r_exec: torch.Tensor,
    r_detect: torch.Tensor,
    feature_families: dict[str, dict[int, list[int]]] | None,
    scope_release: str | None,
    scope_width: str,
    scope_l0: str,
    scope_device: str,
    scope_dtype: str,
    scope_batch_size: int,
) -> dict[str, Any]:
    target_states = prompt_states[TARGET_LAYER]
    late_states = prompt_states[LATE_FAMILY_LAYER]
    out = {
        "z_exec_mean": float(projection_values_1d(target_states, r_exec).mean()),
        "z_detect_mean": float(projection_values_1d(target_states, r_detect).mean()),
        "late_family_means": {},
    }
    if feature_families is None or scope_release is None:
        return out
    sae, _ = load_scope_sae(
        release=scope_release,
        sae_id=build_scope_sae_id(LATE_FAMILY_LAYER, width=scope_width, l0=scope_l0),
        device=scope_device,
        dtype=scope_dtype,
    )
    feature_acts = encode_scope_features(sae, late_states, batch_size=scope_batch_size)
    layer_scores = feature_family_scores(
        feature_acts,
        layer=LATE_FAMILY_LAYER,
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


def summarize_audit_scores(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(row["audit_scores"][key]) for row in rows]
    return {
        "n": len(values),
        "mean": round(sum(values) / max(len(values), 1), 4),
        "max": round(max(values, default=0.0), 4),
        "over_0_5": sum(value >= 0.5 for value in values),
        "over_0_8": sum(value >= 0.8 for value in values),
    }


def summarize_condition_records(
    rows: Sequence[dict[str, Any]],
    *,
    prompt_geometry: dict[str, Any],
) -> dict[str, Any]:
    label_payload = summarize_label_records([row["label"] for row in rows])
    harmful_rows = [row for row in rows if row["group"] == "harmful" and "audit_scores" in row]
    out = {
        "n": len(rows),
        "label_summary": label_payload,
        "prompt_geometry": prompt_geometry,
        "by_source_group": {},
    }
    if harmful_rows:
        out["harmful_risk"] = {
            "dangerous": summarize_audit_scores(harmful_rows, "dangerous"),
            DIRECT_DANGER_FEWSHOT_KEY: summarize_audit_scores(harmful_rows, DIRECT_DANGER_FEWSHOT_KEY),
        }
    for source_group in sorted({row["source_group"] for row in rows}):
        group_rows = [row for row in rows if row["source_group"] == source_group]
        out["by_source_group"][source_group] = summarize_label_records([row["label"] for row in group_rows])
    return out


def build_effect_deltas(condition_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = condition_summaries["baseline"]
    baseline_counts = baseline["label_summary"]["primary_counts"]
    baseline_n = max(int(baseline["label_summary"]["n"]), 1)
    baseline_actionable = int(baseline["label_summary"]["actionable_unsafe"])
    baseline_risk = baseline.get("harmful_risk", {})
    out: dict[str, Any] = {}
    for condition_name, summary in condition_summaries.items():
        if condition_name == "baseline":
            continue
        label_counts = summary["label_summary"]["primary_counts"]
        payload = {
            "delta_actionable_unsafe_rate": round(
                int(summary["label_summary"]["actionable_unsafe"]) / max(int(summary["label_summary"]["n"]), 1)
                - baseline_actionable / baseline_n,
                4,
            ),
            "delta_primary_counts": {
                label: int(label_counts.get(label, 0) - baseline_counts.get(label, 0))
                for label in sorted(set(baseline_counts) | set(label_counts))
            },
            "delta_prompt_geometry": {
                key: round(
                    float(summary["prompt_geometry"].get(key, 0.0))
                    - float(baseline["prompt_geometry"].get(key, 0.0)),
                    4,
                )
                for key in ["z_exec_mean", "z_detect_mean"]
            },
        }
        if baseline_risk and summary.get("harmful_risk"):
            payload["delta_harmful_risk_mean"] = {
                "dangerous": round(
                    float(summary["harmful_risk"]["dangerous"]["mean"])
                    - float(baseline_risk["dangerous"]["mean"]),
                    4,
                ),
                DIRECT_DANGER_FEWSHOT_KEY: round(
                    float(summary["harmful_risk"][DIRECT_DANGER_FEWSHOT_KEY]["mean"])
                    - float(baseline_risk[DIRECT_DANGER_FEWSHOT_KEY]["mean"]),
                    4,
                ),
            }
        out[condition_name] = payload
    return out


def attach_harmful_audit(
    *,
    rows: list[dict[str, Any]],
    auditor: ShieldGemmaAuditor,
    condition_name: str,
    truncate_response: int,
) -> None:
    harmful_indices = [idx for idx, row in enumerate(rows) if row["group"] == "harmful"]
    if not harmful_indices:
        return
    items = [
        AuditItem(
            prompt=rows[idx]["prompt"],
            response=rows[idx]["response"],
            source_file="exp28_detect_family_causal",
            source_path=condition_name,
            meta={"topic": rows[idx]["topic"], "group": rows[idx]["group"]},
        )
        for idx in harmful_indices
    ]
    results = auditor.audit_items(
        items,
        truncate_response=truncate_response,
        progress=f"exp28.shield.{condition_name}",
    )
    for idx, result in zip(harmful_indices, results):
        rows[idx]["audit_scores"] = {
            key: float(score.yes_prob)
            for key, score in result.scores.items()
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topics", default=",".join(DEFAULT_TOPICS))
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--condition_names", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--n_eval_per_group", type=int, default=4)
    parser.add_argument("--n_train_exec", type=int, default=100)
    parser.add_argument("--detect_alpha", type=float, default=4.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--shield_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--shield_truncate", type=int, default=500)
    parser.add_argument("--scope_site", default="res", choices=["res", "att", "mlp", "transcoder"])
    parser.add_argument("--scope_release", default=None)
    parser.add_argument("--scope_width", default="16k")
    parser.add_argument("--scope_l0", default="small")
    parser.add_argument("--scope_device", default="auto")
    parser.add_argument("--scope_dtype", default="float32")
    parser.add_argument("--scope_batch_size", type=int, default=128)
    parser.add_argument("--scope_top_k_family", type=int, default=3)
    parser.add_argument("--exp17_input", default="results/exp17_gemma_scope_feature_probe_full.json")
    parser.add_argument("--output", default="results/exp28_detect_family_causal.json")
    args = parser.parse_args()

    topics = parse_name_list(args.topics)
    groups = parse_name_list(args.groups)
    condition_names = parse_name_list(args.condition_names)

    set_seed(args.seed)
    model, tokenizer = load_model(args.model, args.hf_token)
    device = next(model.parameters()).device

    r_exec = extract_and_cache(
        model,
        tokenizer,
        args.model,
        layers=[TARGET_LAYER],
        n_train=args.n_train_exec,
        seed=args.seed,
    )[TARGET_LAYER].to(device)
    r_detect = collect_direction_candidate(
        model,
        tokenizer,
        r_exec=r_exec,
        n_train=args.n_train_exec,
        seed=args.seed,
    ).to(device)

    topic_payload_eval = load_topic_banks(
        split="test",
        seed=args.seed,
        topics=topics,
        n_per_group=args.n_eval_per_group,
    )
    eval_rows = build_eval_rows(topic_payload_eval, groups=groups)

    resolved_scope_device = args.scope_device
    if resolved_scope_device == "auto":
        resolved_scope_device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_shield_device = args.shield_device
    if resolved_shield_device == "auto":
        resolved_shield_device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_families = None
    scope_release = None
    exp17_input = resolve_runtime_path(args.exp17_input)
    if exp17_input.exists():
        feature_families = load_feature_families(
            exp17_input,
            layers=[LATE_FAMILY_LAYER],
            top_k=args.scope_top_k_family,
        )
        scope_release = args.scope_release or build_scope_release(
            args.model,
            site=args.scope_site,
            all_layers=True,
        )

    auditor = ShieldGemmaAuditor(
        device=resolved_shield_device,
        prompt_specs=default_prompt_specs([HarmType.DANGEROUS]) + [direct_danger_fewshot_spec()],
    )

    condition_rows: dict[str, list[dict[str, Any]]] = {}
    condition_summaries: dict[str, dict[str, Any]] = {}

    for condition_name in condition_names:
        row_results, prompt_states = run_condition_for_rows(
            model,
            tokenizer,
            rows=eval_rows,
            condition_name=condition_name,
            r_exec=r_exec,
            r_detect=r_detect,
            detect_alpha=args.detect_alpha,
            max_new_tokens=args.max_new_tokens,
        )
        attach_harmful_audit(
            rows=row_results,
            auditor=auditor,
            condition_name=condition_name,
            truncate_response=args.shield_truncate,
        )
        prompt_geometry = summarize_prompt_geometry(
            prompt_states,
            r_exec=r_exec,
            r_detect=r_detect,
            feature_families=feature_families,
            scope_release=scope_release,
            scope_width=args.scope_width,
            scope_l0=args.scope_l0,
            scope_device=resolved_scope_device,
            scope_dtype=args.scope_dtype,
            scope_batch_size=args.scope_batch_size,
        )
        for row in row_results:
            row.pop("response", None)
        condition_rows[condition_name] = row_results
        condition_summaries[condition_name] = summarize_condition_records(
            row_results,
            prompt_geometry=prompt_geometry,
        )

    output_path = resolve_runtime_path(args.output, for_output=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "seed": args.seed,
        "topics": topics,
        "groups": groups,
        "condition_names": condition_names,
        "n_eval_rows": len(eval_rows),
        "detect_alpha": args.detect_alpha,
        "direction_summary": {
            "r_exec_norm": float(r_exec.norm().item()),
            "r_detect_norm": float(r_detect.norm().item()),
            "cosine_r_exec_r_detect": float(torch.dot(r_exec, r_detect).item()),
        },
        "scope_release": scope_release,
        "scope_width": args.scope_width,
        "scope_l0": args.scope_l0,
        "scope_device": resolved_scope_device,
        "shield_device": resolved_shield_device,
        "feature_families": {
            family_name: {str(layer): features for layer, features in layer_map.items()}
            for family_name, layer_map in (feature_families or {}).items()
        },
        "condition_summaries": condition_summaries,
        "effect_deltas_vs_baseline": build_effect_deltas(condition_summaries),
        "items": {
            condition_name: [
                {
                    "topic": row["topic"],
                    "group": row["group"],
                    "source_group": row["source_group"],
                    "primary_label": row["label"]["primary_label"],
                    "secondary_flags": row["label"]["secondary_flags"],
                    "actionable_unsafe": row["label"]["actionable_unsafe"],
                    "review_priority": row["label"]["review_priority"],
                    "audit_scores": row.get("audit_scores"),
                }
                for row in rows
            ]
            for condition_name, rows in condition_rows.items()
        },
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp_28] saved={output_path}")


if __name__ == "__main__":
    main()
