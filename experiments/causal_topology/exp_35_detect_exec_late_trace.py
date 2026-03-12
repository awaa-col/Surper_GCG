"""
experiments/causal_topology/exp_35_detect_exec_late_trace.py
============================================================
Run multiple generation-step schedules and compare:
  - z_detect_pure,
  - z_exec,
  - late family trajectories,
  - trajectory labels,
  - risk crossings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from poc.data.topic_banks import load_topic_banks  # noqa: E402
from poc.experiments.exp_19_l17_l23_late_impact import (  # noqa: E402
    LATE_LAYERS,
    TARGET_LAYER,
    load_feature_families,
    load_model,
    projection_values_1d,
)
from poc.experiments.exp_31_generation_step_detect_schedule import (  # noqa: E402
    DEFAULT_TOPICS,
    attach_scope_families,
    build_action_hooks,
    classify_step_label,
    collect_pure_detect_prompt_vector,
    collect_step_forward,
    decode_ids,
    layer_family_mean,
    parse_schedule,
    pick_prompt,
    summarize_steps,
)
from poc.experiments.family_structure.common import (  # noqa: E402
    parse_name_list,
    resolve_runtime_path,
    save_json,
)
from poc.probes.direction_cache import extract_and_cache  # noqa: E402
from poc.probes.extract import _build_prompt  # noqa: E402
from poc.probes.gemma_scope import build_scope_release  # noqa: E402
from poc.probes.response_spans import build_span_records  # noqa: E402
from poc.probes.shield_audit import (  # noqa: E402
    AuditItem,
    HarmType,
    ShieldGemmaAuditor,
    default_prompt_specs,
    direct_danger_fewshot_spec,
)
from poc.probes.stats import set_seed  # noqa: E402


def parse_schedules(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(";") if item.strip()]
    if not items:
        raise ValueError("Expected at least one schedule.")
    return items


def action_name_for_step(schedule: list[dict[str, int | str]], step: int) -> str:
    active = str(schedule[0]["action_name"])
    for phase in schedule:
        if step >= int(phase["start_step"]):
            active = str(phase["action_name"])
        else:
            break
    return active


def build_trace_for_schedule(
    *,
    model,
    tokenizer,
    prompt: str,
    schedule_raw: str,
    r_exec,
    r_detect_pure,
    auditor,
    detect_alpha: float,
    max_new_tokens: int,
    audit_every: int,
):
    schedule = parse_schedule(schedule_raw)
    device = next(model.parameters()).device
    prompt_text = _build_prompt(tokenizer, prompt)
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    generated_ids: list[int] = []
    steps: list[dict[str, Any]] = []

    step_iter = tqdm(
        range(max_new_tokens),
        desc=f"exp35.trace.{schedule_raw}",
        leave=False,
    )
    for step_index in step_iter:
        action_name = action_name_for_step(schedule, step_index)
        hook_specs = build_action_hooks(
            action_name=action_name,
            r_exec=r_exec,
            r_detect_pure=r_detect_pure,
            detect_alpha=detect_alpha,
        )
        current_ids = input_ids
        if generated_ids:
            current_ids = torch.cat(
                [input_ids, torch.tensor([generated_ids], device=device, dtype=input_ids.dtype)],
                dim=1,
            )

        logits, states = collect_step_forward(
            model,
            input_ids=current_ids,
            hook_specs=hook_specs,
            capture_layers=[TARGET_LAYER] + LATE_LAYERS,
        )
        next_token_id = int(torch.argmax(logits).item())
        before_text = decode_ids(tokenizer, generated_ids)
        generated_ids.append(next_token_id)
        after_text = decode_ids(tokenizer, generated_ids)

        record = {
            "step": int(step_index),
            "action_name": action_name,
            "assistant_text_before": before_text,
            "assistant_text_after": after_text,
            "generated_token_id": next_token_id,
            "generated_token_text": decode_ids(tokenizer, [next_token_id]),
            "z_exec": float(projection_values_1d(states[TARGET_LAYER].unsqueeze(0), r_exec)[0]),
            "z_detect_pure": float(projection_values_1d(states[TARGET_LAYER].unsqueeze(0), r_detect_pure)[0]),
            "states": states,
        }
        steps.append(record)

        if audit_every <= 1 or step_index % audit_every == 0 or step_index == max_new_tokens - 1:
            audit_item = AuditItem(
                prompt=prompt,
                response=after_text,
                source_file="exp35_detect_exec_late_trace",
                source_path=f"{schedule_raw}.steps[{step_index}]",
            )
            audit_result = auditor.audit_item(audit_item, index=step_index + 1, truncate_response=500)
            span_records = build_span_records([prompt], [after_text], f"trace:{schedule_raw}")
            span_labels = [span["label"] for span in span_records[0]["spans"]]
            record["audit_scores"] = audit_result.yes_scores()
            record["span_labels"] = span_labels
            record["trajectory_label"] = classify_step_label(
                span_labels,
                record["audit_scores"].get("dangerous"),
            )
        else:
            record["audit_scores"] = {}
            record["span_labels"] = []
            record["trajectory_label"] = "other"

        step_iter.set_postfix_str(
            f"step={step_index} action={action_name} token={record['generated_token_text'][:16]!r}"
        )

    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topics", default=",".join(DEFAULT_TOPICS))
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt_topic", default="explosives")
    parser.add_argument("--prompt_group", default="harmful", choices=["harmful", "supportive", "safe_info"])
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--schedules", default="baseline@0;pure_add@0;pure_sub@0;exec_off@0;exec_off@0,pure_add@1")
    parser.add_argument("--detect_alpha", type=float, default=4.0)
    parser.add_argument("--n_train_per_group", type=int, default=6)
    parser.add_argument("--n_train_exec", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--audit_every", type=int, default=1)
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
    parser.add_argument("--output", default="results/causal_topology/exp35_detect_exec_late_trace.json")
    args = parser.parse_args()

    set_seed(args.seed)
    topics = parse_name_list(args.topics)
    schedules = parse_schedules(args.schedules)

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
    print("[exp_35] collecting pure detect vector")
    topic_payload_train = load_topic_banks(
        split="train",
        seed=args.seed,
        topics=topics,
        n_per_group=args.n_train_per_group,
    )
    r_detect_pure = collect_pure_detect_prompt_vector(
        model,
        tokenizer,
        topic_payload_train=topic_payload_train,
        r_exec=r_exec,
    )

    topic_payload_eval = load_topic_banks(
        split="test",
        seed=args.seed,
        topics=topics,
        n_per_group=4,
    )
    prompt = args.prompt or pick_prompt(
        topic_payload_eval,
        topic=args.prompt_topic,
        group=args.prompt_group,
        index=args.prompt_index,
    )

    resolved_scope_device = args.scope_device
    if resolved_scope_device == "auto":
        resolved_scope_device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_shield_device = args.shield_device
    if resolved_shield_device == "auto":
        resolved_shield_device = "cuda" if torch.cuda.is_available() else "cpu"

    exp17_input = resolve_runtime_path(args.exp17_input)
    feature_families = None
    scope_release = None
    if exp17_input.exists():
        feature_families = load_feature_families(
            exp17_input,
            layers=LATE_LAYERS,
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
    print(f"[exp_35] prompt_topic={args.prompt_topic} prompt_group={args.prompt_group} n_schedules={len(schedules)}")

    schedule_payloads: dict[str, Any] = {}
    for schedule_raw in tqdm(schedules, desc="exp35.schedules"):
        print(f"[exp_35] running schedule={schedule_raw}")
        steps = build_trace_for_schedule(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            schedule_raw=schedule_raw,
            r_exec=r_exec,
            r_detect_pure=r_detect_pure,
            auditor=auditor,
            detect_alpha=args.detect_alpha,
            max_new_tokens=args.max_new_tokens,
            audit_every=args.audit_every,
        )
        print(f"[exp_35] attaching scope families schedule={schedule_raw}")
        scope_infos = attach_scope_families(
            steps=steps,
            feature_families=feature_families,
            scope_release=scope_release,
            scope_width=args.scope_width,
            scope_l0=args.scope_l0,
            scope_device=resolved_scope_device,
            scope_dtype=args.scope_dtype,
            scope_batch_size=args.scope_batch_size,
        )
        summary = summarize_steps(steps)
        summary["z_detect_min"] = float(min(step["z_detect_pure"] for step in steps))
        summary["z_detect_max"] = float(max(step["z_detect_pure"] for step in steps))
        summary["z_exec_mean"] = float(sum(step["z_exec"] for step in steps) / max(1, len(steps)))
        summary["final_safe_response_family"] = layer_family_mean(steps[-1], "safe_response_family")
        summary["final_unsafe_exec_family"] = layer_family_mean(steps[-1], "unsafe_exec_family")
        schedule_payloads[schedule_raw] = {
            "summary": summary,
            "scope_infos": scope_infos,
            "steps": steps,
        }

    payload = {
        "model": args.model,
        "seed": args.seed,
        "topics": topics,
        "prompt": prompt,
        "prompt_topic": args.prompt_topic,
        "prompt_group": args.prompt_group,
        "detect_alpha": args.detect_alpha,
        "schedules": schedules,
        "schedule_payloads": schedule_payloads,
    }
    saved = save_json(payload, args.output)
    print(f"[exp_35] saved={saved}")


if __name__ == "__main__":
    main()
