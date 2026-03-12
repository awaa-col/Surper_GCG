"""
experiments/family_structure/exp_34_family_dependency_matrix.py
===============================================================
Prompt-time dependency matrix between late safety families.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

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
)
from poc.probes.extract import _build_prompt  # noqa: E402
from poc.probes.gemma_scope import (  # noqa: E402
    build_scope_release,
    build_scope_sae_id,
    encode_scope_features,
    load_scope_sae,
    make_scope_feature_edit_hook,
)
from poc.probes.stats import set_seed  # noqa: E402


DEFAULT_TOPICS = ["self_harm", "explosives", "fraud", "hate_or_abuse"]
DEFAULT_GROUPS = ["harmful", "supportive", "safe_info"]


def mean_scores(
    saes_by_layer: dict[int, object],
    captured_states: dict[int, list[torch.Tensor]],
    *,
    feature_families: dict[str, dict[int, list[int]]],
    batch_size: int,
) -> dict[str, float]:
    sums: dict[str, list[float]] = defaultdict(list)
    for layer, state_list in captured_states.items():
        states = torch.stack(state_list, dim=0)
        feature_acts = encode_scope_features(
            saes_by_layer[layer],
            states,
            batch_size=batch_size,
        )
        layer_scores = feature_family_scores(
            feature_acts,
            layer=layer,
            feature_families=feature_families,
        )
        for family_name, values in layer_scores.items():
            sums[family_name].append(float(values.mean()))
    return {family_name: float(sum(values) / len(values)) for family_name, values in sums.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topics", default=",".join(DEFAULT_TOPICS))
    parser.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--n_per_group", type=int, default=4)
    parser.add_argument("--family_names", default="refusal_family,risk_family,empathy_family,resource_family,unsafe_exec_family,safe_response_family")
    parser.add_argument("--late_layers", default="21,22,23,24")
    parser.add_argument("--scope_site", default="res", choices=["res", "att", "mlp", "transcoder"])
    parser.add_argument("--scope_release", default=None)
    parser.add_argument("--scope_width", default="16k")
    parser.add_argument("--scope_l0", default="small")
    parser.add_argument("--scope_device", default="auto")
    parser.add_argument("--scope_dtype", default="float32")
    parser.add_argument("--scope_batch_size", type=int, default=128)
    parser.add_argument("--scope_top_k_family", type=int, default=3)
    parser.add_argument("--exp17_input", default="results/exp17_gemma_scope_feature_probe_full.json")
    parser.add_argument("--output", default="results/family_structure/exp34_family_dependency_matrix.json")
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

    topic_payload = load_topic_banks(
        split=args.split,
        seed=args.seed,
        topics=topics,
        n_per_group=args.n_per_group,
    )
    rows = build_eval_rows(topic_payload, groups=groups)

    exp17_input = resolve_runtime_path(args.exp17_input)
    feature_families = load_feature_families(
        exp17_input,
        layers=LATE_LAYERS,
        top_k=args.scope_top_k_family,
    )
    scope_release = args.scope_release or build_scope_release(args.model, site=args.scope_site, all_layers=True)

    saes_by_layer: dict[int, object] = {}
    for layer in late_layers:
        sae_id = build_scope_sae_id(layer, width=args.scope_width, l0=args.scope_l0)
        sae, _ = load_scope_sae(
            release=scope_release,
            sae_id=sae_id,
            device=resolved_scope_device,
            dtype=args.scope_dtype,
        )
        saes_by_layer[layer] = sae

    baseline_captured: dict[int, list[torch.Tensor]] = {layer: [] for layer in late_layers}
    for row in tqdm(rows, desc="exp34.baseline"):
        text = _build_prompt(tokenizer, row["prompt"])
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model_device)
        holder: dict[int, torch.Tensor] = {}
        handles = []
        try:
            for layer in late_layers:
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
        for layer in late_layers:
            baseline_captured[layer].append(holder[layer])

    baseline_means = mean_scores(
        saes_by_layer,
        baseline_captured,
        feature_families=feature_families,
        batch_size=args.scope_batch_size,
    )

    dependency: dict[str, dict] = {}
    for source_family in family_names:
        captured: dict[int, list[torch.Tensor]] = {layer: [] for layer in late_layers}
        for row in tqdm(rows, desc=f"exp34.{source_family}"):
            text = _build_prompt(tokenizer, row["prompt"])
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model_device)
            holder: dict[int, torch.Tensor] = {}
            handles = []
            try:
                for layer in late_layers:
                    source_features = feature_families.get(source_family, {}).get(layer, [])
                    if source_features:
                        handles.append(
                            model.model.layers[layer].register_forward_hook(
                                make_scope_feature_edit_hook(
                                    saes_by_layer[layer],
                                    features=source_features,
                                    mode="zero",
                                )
                            )
                        )
                for layer in late_layers:
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
            for layer in late_layers:
                captured[layer].append(holder[layer])

        edited_means = mean_scores(
            saes_by_layer,
            captured,
            feature_families=feature_families,
            batch_size=args.scope_batch_size,
        )
        dependency[source_family] = {
            "edited_means": edited_means,
            "delta_from_baseline": {
                target_family: float(edited_means.get(target_family, 0.0) - baseline_means.get(target_family, 0.0))
                for target_family in sorted(set(baseline_means.keys()) | set(edited_means.keys()))
            },
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
        "late_layers": late_layers,
        "baseline_means": baseline_means,
        "dependency": dependency,
    }
    saved = save_json(payload, args.output)
    print(f"[exp_34] saved={saved}")

    for sae in saes_by_layer.values():
        try:
            sae.to("cpu")
        except Exception:
            pass


if __name__ == "__main__":
    main()
