"""
experiments/exp_22_standard_soft_reply_token_probe.py
=====================================================
Run Exp21-style progressive assistant-prefill analysis on the standard soft-reply
examples collected in `标准软道歉类型.md`.

Gemma-3-1B-IT remains the mechanism target.
ShieldGemma is used only as the external safety auditor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.exp_19_l17_l23_late_impact import (  # noqa: E402
    LATE_LAYERS,
    TARGET_LAYER,
    build_condition_hooks,
    build_scope_release,
    collect_direction_candidate,
    load_feature_families,
    load_model,
    run_shield_audit,
)
from experiments.exp_21_dangerous_reply_token_probe import (  # noqa: E402
    DEFAULT_CHECKPOINT_MODE,
    DEFAULT_PROMPT,
    attach_scope_family_traces_all,
    attach_shield_metadata,
    attach_span_metadata,
    build_prefix_entries,
    classify_trajectory_label,
    clear_torch_memory,
    collect_checkpoint_probes,
    generate_continuations,
    parse_int_list,
    parse_name_list,
    resolve_checkpoint_counts,
    resolve_runtime_path,
    summarize_checkpoint_geometry,
    summarize_trajectory_sequence,
)
from probes.direction_cache import extract_and_cache  # noqa: E402
from probes.response_spans import build_span_records, summarize_span_records  # noqa: E402
from probes.shield_audit import ShieldGemmaAuditor  # noqa: E402
from probes.stats import set_seed  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_FILE = ROOT / "标准软道歉类型.md"
DEFAULT_SAMPLE_IDS = [
    "soft_apology",
    "safety_statement_soft_apology",
    "disclaimer_dangerous_content",
    "dan",
]
DEFAULT_CONDITIONS = ["baseline", "l23_only", "l17_l23"]

STANDARD_SAMPLE_ID_MAP = {
    "软道歉": "soft_apology",
    "安全声明+软道歉": "safety_statement_soft_apology",
    "免责声明+危险内容": "disclaimer_dangerous_content",
    "DAN": "dan",
}


def parse_standard_soft_reply_samples(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^##\s+\d+\.\s+(?P<label>.+?)\s*$.*?^```text\s*\n(?P<body>.*?)\n```",
        re.MULTILINE | re.DOTALL,
    )
    samples: list[dict] = []
    for match in pattern.finditer(text):
        label = match.group("label").strip()
        body = match.group("body").strip()
        sample_id = STANDARD_SAMPLE_ID_MAP.get(label)
        if sample_id is None:
            continue
        samples.append(
            {
                "sample_id": sample_id,
                "label": label,
                "text": body,
            }
        )
    if not samples:
        raise ValueError(f"No standard soft-reply samples found in {path}")
    return samples


def sample_comparison_entry(payload: dict) -> dict:
    trajectory = payload.get("trajectory_summary") or {}
    shield_summary = ((payload.get("shield_audit") or {}).get("summary") or {})
    shield_max = shield_summary.get("max_yes_prob") or {}
    return {
        "counts": trajectory.get("counts"),
        "first_prefix_by_label": trajectory.get("first_prefix_by_label"),
        "danger_token_lock_prefix_over_0_5": trajectory.get("danger_token_lock_prefix_over_0_5"),
        "danger_token_lock_prefix_over_0_9": trajectory.get("danger_token_lock_prefix_over_0_9"),
        "shield_risk_prefix_over_0_5": trajectory.get("shield_risk_prefix_over_0_5"),
        "shield_risk_prefix_over_0_8": trajectory.get("shield_risk_prefix_over_0_8"),
        "shield_max_yes_prob": shield_max,
    }


def build_comparison_summary(samples_payload: dict[str, dict]) -> dict:
    comparison: dict[str, dict] = {}
    for sample_id, sample_payload in samples_payload.items():
        comparison[sample_id] = {
            condition_name: sample_comparison_entry(condition_payload)
            for condition_name, condition_payload in sample_payload["conditions"].items()
        }
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--sample_path", default=str(DEFAULT_SAMPLE_FILE))
    parser.add_argument("--sample_names", default=",".join(DEFAULT_SAMPLE_IDS))
    parser.add_argument("--topic_label", default="standard_soft_reply_probe")
    parser.add_argument("--condition_names", default="baseline,l23_only,l17_l23")
    parser.add_argument("--checkpoint_tokens", default="")
    parser.add_argument(
        "--checkpoint_mode",
        default=DEFAULT_CHECKPOINT_MODE,
        choices=["coarse", "fine_early", "dense"],
    )
    parser.add_argument("--top_logit_k", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--shield_truncate", type=int, default=500)
    parser.add_argument("--skip_shield_audit", action="store_true")
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
    parser.add_argument("--output", default="results/exp22_standard_soft_reply_token_probe.json")
    args = parser.parse_args()

    set_seed(args.seed)

    condition_names = parse_name_list(
        args.condition_names,
        allowed=["baseline", "l23_only", "l17_l23", "dual_l17_l23"],
    )
    sample_names = parse_name_list(args.sample_names, allowed=DEFAULT_SAMPLE_IDS)
    requested_checkpoints = (
        parse_int_list(args.checkpoint_tokens) if args.checkpoint_tokens.strip() else None
    )

    sample_path = resolve_runtime_path(args.sample_path)
    sample_specs = parse_standard_soft_reply_samples(sample_path)
    sample_specs = [sample for sample in sample_specs if sample["sample_id"] in sample_names]

    model, tokenizer = load_model(args.model, args.hf_token)

    r_exec_dict = extract_and_cache(
        model,
        tokenizer,
        layers=[TARGET_LAYER],
        n_train=args.n_train_exec,
        seed=args.seed,
        model_name=args.model,
    )
    r_exec = r_exec_dict[TARGET_LAYER]

    r_l23_dict = extract_and_cache(
        model,
        tokenizer,
        layers=[23],
        n_train=args.n_train_exec,
        seed=args.seed,
        model_name=args.model,
    )
    r_l23 = r_l23_dict[23]

    r_soft = collect_direction_candidate(
        model,
        tokenizer,
        r_exec=r_exec,
        n_train=args.n_train_exec,
        seed=args.seed,
    )

    feature_families = load_feature_families(
        resolve_runtime_path(args.exp17_input),
        layers=LATE_LAYERS,
        top_k=args.scope_top_k_family,
    )
    scope_release = args.scope_release or build_scope_release(args.model, site=args.scope_site)

    resolved_scope_device = args.scope_device
    if resolved_scope_device == "auto":
        resolved_scope_device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_shield_device = args.shield_device
    if resolved_shield_device == "auto":
        resolved_shield_device = "cuda" if torch.cuda.is_available() else "cpu"

    samples_payload: dict[str, dict] = {}
    global_scope_infos: dict[int, dict] = {}
    capture_layers = [TARGET_LAYER, 23] + LATE_LAYERS

    for sample_spec in sample_specs:
        sample_text = sample_spec["text"]
        sample_token_ids = tokenizer.encode(sample_text, add_special_tokens=False)
        prefix_entries, sample_token_ids = build_prefix_entries(
            tokenizer,
            sample_text,
            checkpoint_counts=resolve_checkpoint_counts(
                token_count=len(sample_token_ids),
                requested=requested_checkpoints,
                mode=args.checkpoint_mode,
            ),
        )

        prompt_rows = [
            {"topic": args.topic_label, "prompt": args.prompt}
            for _ in prefix_entries
        ]

        conditions_payload: dict[str, dict] = {}
        for condition_name in condition_names:
            hook_specs = build_condition_hooks(
                condition_name,
                r_exec=r_exec,
                r_l23=r_l23,
            )
            checkpoints = collect_checkpoint_probes(
                model,
                tokenizer,
                prompt=args.prompt,
                entries=prefix_entries,
                hook_specs=hook_specs,
                capture_layers=capture_layers,
                top_logit_k=args.top_logit_k,
                desc=f"exp22.probe.{sample_spec['sample_id']}.{condition_name}",
            )
            checkpoints, _ = summarize_checkpoint_geometry(
                checkpoints=checkpoints,
                capture_layers=capture_layers,
                r_exec=r_exec,
                r_l23=r_l23,
                r_soft=r_soft,
            )

            continuations = generate_continuations(
                model,
                tokenizer,
                prompt=args.prompt,
                checkpoints=checkpoints,
                hook_specs=hook_specs,
                max_new_tokens=args.max_new_tokens,
                desc=f"exp22.gen.{sample_spec['sample_id']}.{condition_name}",
            )

            full_assistant_texts = []
            for item, continuation in zip(checkpoints, continuations):
                item["continuation"] = continuation
                item["continuation_preview"] = continuation[:240]
                item["full_assistant_text"] = item["prefix_text"] + continuation
                full_assistant_texts.append(item["full_assistant_text"])

            span_records = build_span_records(
                [args.prompt] * len(full_assistant_texts),
                full_assistant_texts,
                source_group="harmful_exec_only",
            )
            attach_span_metadata(checkpoints, span_records)
            span_summary = summarize_span_records(span_records)

            conditions_payload[condition_name] = {
                "n_checkpoints": len(checkpoints),
                "checkpoints": checkpoints,
                "span_summary": span_summary,
                "shield_audit": None,
                "trajectory_summary": None,
            }

        scope_infos = attach_scope_family_traces_all(
            conditions_payload=conditions_payload,
            feature_families=feature_families,
            scope_release=scope_release,
            scope_width=args.scope_width,
            scope_l0=args.scope_l0,
            scope_device=resolved_scope_device,
            scope_dtype=args.scope_dtype,
            batch_size=args.scope_batch_size,
        )
        global_scope_infos.update(scope_infos)
        clear_torch_memory()

        samples_payload[sample_spec["sample_id"]] = {
            "sample_id": sample_spec["sample_id"],
            "label": sample_spec["label"],
            "sample_preview": sample_text[:500],
            "sample_token_count": len(sample_token_ids),
            "checkpoint_token_counts": [int(item["prefix_tokens"]) for item in prefix_entries],
            "conditions": conditions_payload,
        }

    del model
    clear_torch_memory()

    shield_auditor = None
    if not args.skip_shield_audit:
        shield_auditor = ShieldGemmaAuditor(device=resolved_shield_device)

    for sample_id, sample_payload in samples_payload.items():
        for condition_name, condition_payload in sample_payload["conditions"].items():
            checkpoints = condition_payload["checkpoints"]
            full_assistant_texts = [item["full_assistant_text"] for item in checkpoints]
            shield_summary = None
            if shield_auditor is not None:
                prompt_rows = [
                    {"topic": args.topic_label, "prompt": args.prompt}
                    for _ in checkpoints
                ]
                shield_summary = run_shield_audit(
                    prompt_rows,
                    full_assistant_texts,
                    source_path=f"exp22.{sample_id}.{condition_name}",
                    truncate_response=args.shield_truncate,
                    auditor=shield_auditor,
                )
            attach_shield_metadata(checkpoints, shield_summary)
            for item in checkpoints:
                item["trajectory_label"] = classify_trajectory_label(item)
            condition_payload["shield_audit"] = shield_summary
            condition_payload["trajectory_summary"] = summarize_trajectory_sequence(checkpoints)

    if shield_auditor is not None:
        shield_auditor.unload()

    payload = {
        "model": args.model,
        "seed": args.seed,
        "prompt": args.prompt,
        "topic_label": args.topic_label,
        "sample_path": str(sample_path),
        "sample_names": [sample["sample_id"] for sample in sample_specs],
        "condition_names": condition_names,
        "checkpoint_mode": args.checkpoint_mode,
        "direction_summary": {
            "r_exec_norm": float(torch.norm(r_exec).item()),
            "r_l23_norm": float(torch.norm(r_l23).item()),
            "r_soft_safe_style_candidate_norm": float(torch.norm(r_soft).item()),
            "cosine_r_exec_r_l23": float(torch.dot(r_exec, r_l23).item()),
            "cosine_r_exec_r_soft_safe_style_candidate": float(torch.dot(r_exec, r_soft).item()),
        },
        "scope_site": args.scope_site,
        "scope_release": scope_release,
        "scope_width": args.scope_width,
        "scope_l0": args.scope_l0,
        "scope_device": resolved_scope_device,
        "shield_device": resolved_shield_device,
        "scope_infos": {
            str(layer): info
            for layer, info in global_scope_infos.items()
        },
        "feature_families": {
            family_name: {
                str(layer): features
                for layer, features in sorted(layer_map.items())
            }
            for family_name, layer_map in feature_families.items()
        },
        "samples": samples_payload,
        "comparison_summary": build_comparison_summary(samples_payload),
    }

    output_path = resolve_runtime_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[exp_22] saved={output_path}")


if __name__ == "__main__":
    main()
