from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[3]
POC_ROOT = ROOT / "poc"


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


def save_json(payload: dict[str, Any], output_path: str) -> Path:
    resolved = resolve_runtime_path(output_path, for_output=True)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return resolved


def stack_capture_rows(
    model,
    tokenizer,
    *,
    rows: Sequence[dict[str, str]],
    capture_layers: Sequence[int],
):
    from probes.extract import _build_prompt

    device = next(model.parameters()).device
    state_lists: dict[int, list[torch.Tensor]] = {layer: [] for layer in capture_layers}

    for row in rows:
        holder: dict[int, torch.Tensor] = {}
        text = _build_prompt(tokenizer, row["prompt"])
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        handles = []
        try:
            for layer in capture_layers:
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
                model(input_ids=input_ids, output_hidden_states=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()

        for layer in capture_layers:
            state_lists[layer].append(holder[layer])

    return {layer: torch.stack(items, dim=0) for layer, items in state_lists.items()}
