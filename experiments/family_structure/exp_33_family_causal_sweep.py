"""
experiments/family_structure/exp_33_family_causal_sweep.py
==========================================================
Single-family causal sweep in Gemma Scope feature space.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from poc.data.topic_banks import load_topic_banks  # noqa: E402
from poc.experiments.exp_19_l17_l23_late_impact import (  # noqa: E402
    LATE_LAYERS,
    TARGET_LAYER,
    feature_family_scores,
    load_feature_families,
    load_model,
    projection_values_1d,
)
from poc.experiments.family_structure.common import (  # noqa: E402
    build_eval_rows,
    deterministic_generate,
    parse_name_list,
    resolve_runtime_path,
    save_json,
)
from poc.probes.direction_cache import extract_and_cache  # noqa: E402
from poc.probes.extract import _build_prompt  # noqa: E402
from poc.probes.gemma_scope import (  # noqa: E402
    build_scope_release,
    build_scope_sae_id,
    encode_scope_features,
    load_scope_sae,
    make_scope_feature_edit_hook,
)
from poc.probes.review_labels import label_response, summarize_label_records  # noqa: E402
from poc.probes.shield_audit import (  # noqa: E402
    DIRECT_DANGER_FEWSHOT_KEY,
    AuditItem,
    HarmType,
    ShieldGemmaAuditor,
    default_prompt_specs,
    direct_danger_fewshot_spec,
)
from poc.probes.stats import set_seed  # noqa: E402


DEFAULT_TOPICS = ["self_harm", "explosives", "fraud", "hate_or_abuse"]
DEFAULT_GROUPS = ["harmful", "supportive", "safe_info"]


def build_condition_specs(family_names: Sequence[str], *, add_alpha: float) -> list[dict[str, Any]]:
    specs = [{"name": "baseline", "family_name": None, "mode": None, "alpha": 0.0}]
    for family_name in family_names:
        specs.append({"name": f"{family_name}_sub", "family_name": family_name, "mode": "zero", "alpha": 0.0})
        specs.append({"name": f"{family_name}_add", "family_name": family_name, "mode": "add", "alpha": add_alpha})
    return specs


def collect_family_prompt_scores(
    saes_by_layer: dict[int, object],
    captured_states: dict[int, torch.Tensor],
    *,
    feature_families: dict[str, dict[int, list[int]]],
    batch_size: int,
) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for layer, state in captured_states.items():
        feature_acts = encode_scope_features(
            saes_by_layer[layer],
            state.unsqueeze(0),
            batch_size=batch_size,
        )
        layer_scores = feature_family_scores(
            feature_acts,
            layer=layer,
            feature_families=feature_families,
        )
        for family_name, values in layer_scores.items():
            out.setdefault(family_name, []).append(float(values[0]))
    return {family_name: float(sum(values) / len(values)) for family_name, values in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topics", default=",".join(DEFAULT_TOPICS))
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--n_per_group", type=int, default=4)
    parser.add_argument("--family_names", default="refusal_family,risk_family,empathy_family,resource_family,unsafe_exec_family")
    parser.add_argument("--late_layers", default="21,22,23,24")
    parser.add_argument("--add_alpha", type=float, default=2.0)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--shield_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--scope_site", default="res", choices=["res", "att", "mlp", "transcoder"])
    parser.add_argument("--scope_release", default=None)
    parser.add_argument("--scope_width", default="16k")
    parser.add_argument("--scope_l0", default="small")
    parser.add_argument("--scope_device", default="auto")
    parser.add_argument("--scope_dtype", default="float32")
    parser.add_argument("--scope_batch_size", type=int, default=128)
    parser.add_argument("--scope_top_k_family", type=int, default=3)
    parser.add_argument("--n_train_exec", type=int, default=100)
    parser.add_argument("--exp17_input", default="results/exp17_gemma_scope_feature_probe_full.json")
    parser.add_argument("--output", default="results/family_structure/exp33_family_causal_sweep.json")
    args = parser.parse_args()

    set_seed(args.seed)
    topics = parse_name_list(args.topics)
    groups = parse_name_list(args.groups)
    family_names = parse_name_list(args.family_names)
    late_layers = [int(item.strip()) for item in args.late_layers.split(",") if item.strip()]

    model, tokenizer = load_model(args.model, args.hf_token)
    model_device = next(model.parameters()).device
    resolved_scope_device = args.scope_device
    if resolved_scope_device == "auto":
        resolved_scope_device = str(model_device)
    resolved_shield_device = args.shield_device
    if resolved_shield_device == "auto":
        resolved_shield_device = "cuda" if torch.cuda.is_available() else "cpu"

    exp17_input = resolve_runtime_path(args.exp17_input)
    feature_families = load_feature_families(
        exp17_input,
        layers=LATE_LAYERS,
        top_k=args.scope_top_k_family,
    )
    scope_release = args.scope_release or build_scope_release(args.model, site=args.scope_site, all_layers=True)

    saes_by_layer: dict[int, object] = {}
    scope_infos: dict[str, dict] = {}
    for layer in late_layers:
        sae_id = build_scope_sae_id(layer, width=args.scope_width, l0=args.scope_l0)
        sae, info = load_scope_sae(
            release=scope_release,
            sae_id=sae_id,
            device=resolved_scope_device,
            dtype=args.scope_dtype,
        )
        saes_by_layer[layer] = sae
        scope_infos[str(layer)] = info.to_dict()

    r_exec = extract_and_cache(
        model,
        tokenizer,
        args.model,
        layers=[TARGET_LAYER],
        n_train=args.n_train_exec,
        seed=args.seed,
    )[TARGET_LAYER].to(model_device)

    topic_payload = load_topic_banks(
        split=args.split,
        seed=args.seed,
        topics=topics,
        n_per_group=args.n_per_group,
    )
    rows = build_eval_rows(topic_payload, groups=groups)

    auditor = ShieldGemmaAuditor(
        device=resolved_shield_device,
        prompt_specs=default_prompt_specs([HarmType.DANGEROUS]) + [direct_danger_fewshot_spec()],
    )

    condition_specs = build_condition_specs(family_names, add_alpha=args.add_alpha)
    condition_results: dict[str, Any] = {}

    for condition in condition_specs:
        name = str(condition["name"])
        family_name = condition["family_name"]
        mode = condition["mode"]
        alpha = float(condition["alpha"])
        row_results: list[dict[str, Any]] = []

        for row in tqdm(rows, desc=f"exp33.{name}"):
            text = _build_prompt(tokenizer, row["prompt"])
            inputs = tokenizer(text, return_tensors="pt").to(model_device)
            captured_states: dict[int, torch.Tensor] = {}
            handles = []
            try:
                if family_name is not None:
                    for layer in late_layers:
                        features = feature_families.get(family_name, {}).get(layer, [])
                        if features:
                            handles.append(
                                model.model.layers[layer].register_forward_hook(
                                    make_scope_feature_edit_hook(
                                        saes_by_layer[layer],
                                        features=features,
                                        mode=str(mode),
                                        alpha=alpha,
                                    )
                                )
                            )

                for layer in [TARGET_LAYER] + late_layers:
                    def make_capture(target_layer: int):
                        def capture_hook(module, inputs, output):
                            hidden = output[0] if isinstance(output, tuple) else output
                            captured_states[target_layer] = hidden[0, -1, :].float().cpu()
                            return output

                        return capture_hook

                    handles.append(model.model.layers[layer].register_forward_hook(make_capture(layer)))

                with torch.no_grad():
                    out = deterministic_generate(model, **inputs, max_new_tokens=args.max_new_tokens)
                new_tokens = out[0, inputs["input_ids"].shape[1] :]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            finally:
                for handle in handles:
                    handle.remove()

            audit_item = AuditItem(
                prompt=row["prompt"],
                response=response,
                source_file="exp33_family_causal_sweep",
                source_path=name,
                meta={"topic": row["topic"], "group": row["group"]},
            )
            audit_result = auditor.audit_item(audit_item, truncate_response=500)
            family_prompt_scores = collect_family_prompt_scores(
                saes_by_layer,
                {layer: captured_states[layer] for layer in late_layers},
                feature_families=feature_families,
                batch_size=args.scope_batch_size,
            )
            row_results.append(
                {
                    "topic": row["topic"],
                    "group": row["group"],
                    "source_group": row["source_group"],
                    "prompt": row["prompt"],
                    "response": response,
                    "label": label_response(response, prompt=row["prompt"]),
                    "audit_scores": audit_result.yes_scores(),
                    "z_exec": float(projection_values_1d(captured_states[TARGET_LAYER].unsqueeze(0), r_exec)[0]),
                    "late_family_prompt_scores": family_prompt_scores,
                }
            )

        all_family_names = sorted(
            {
                item_name
                for item in row_results
                for item_name in item["late_family_prompt_scores"].keys()
            }
        )
        condition_results[name] = {
            "family_name": family_name,
            "mode": mode,
            "alpha": alpha,
            "label_summary": summarize_label_records([item["label"] for item in row_results]),
            "dangerous_mean": float(sum(item["audit_scores"].get("dangerous", 0.0) for item in row_results) / max(1, len(row_results))),
            "direct_danger_mean": float(
                sum(item["audit_scores"].get(DIRECT_DANGER_FEWSHOT_KEY, 0.0) for item in row_results) / max(1, len(row_results))
            ),
            "z_exec_mean": float(sum(item["z_exec"] for item in row_results) / max(1, len(row_results))),
            "late_family_prompt_mean": {
                family: float(sum(item["late_family_prompt_scores"].get(family, 0.0) for item in row_results) / max(1, len(row_results)))
                for family in all_family_names
            },
            "rows": row_results,
        }

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
        "scope_infos": scope_infos,
        "condition_results": condition_results,
    }
    saved = save_json(payload, args.output)
    print(f"[exp_33] saved={saved}")

    for sae in saes_by_layer.values():
        try:
            sae.to("cpu")
        except Exception:
            pass


if __name__ == "__main__":
    main()
