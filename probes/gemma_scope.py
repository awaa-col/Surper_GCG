from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch
from tqdm import tqdm

try:
    from sae_lens import SAE
except ImportError as exc:  # pragma: no cover - import guard for optional dependency
    SAE = None
    SAE_IMPORT_ERROR = exc
else:
    SAE_IMPORT_ERROR = None


DEFAULT_SCOPE_WIDTH = "16k"
DEFAULT_SCOPE_L0 = "small"


@dataclass
class ScopeSAEInfo:
    release: str
    sae_id: str
    d_in: int
    d_sae: int
    model_name: str | None
    hook_name: str | None
    hf_hook_name: str | None
    sparsity: float | None

    def to_dict(self) -> dict:
        return {
            "release": self.release,
            "sae_id": self.sae_id,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hf_hook_name": self.hf_hook_name,
            "sparsity": self.sparsity,
        }


def _require_sae_lens() -> None:
    if SAE is None:
        raise RuntimeError(
            "sae-lens is required for Gemma Scope integration. "
            "Install it in the current venv before using this module."
        ) from SAE_IMPORT_ERROR


def build_scope_release(
    model_name: str,
    *,
    site: str = "res",
    all_layers: bool = True,
) -> str:
    model_stub = model_name.split("/")[-1]
    if not model_stub.startswith("gemma-3-"):
        raise ValueError(f"Unsupported model for Gemma Scope 2: {model_name}")

    variant = model_stub.removeprefix("gemma-3-")
    site_alias = {
        "res": "res",
        "att": "att",
        "mlp": "mlp",
        "transcoder": "transcoders",
    }
    if site not in site_alias:
        raise ValueError(f"Unsupported scope site: {site}")

    suffix = "-all" if all_layers else ""
    return f"gemma-scope-2-{variant}-{site_alias[site]}{suffix}"


def build_scope_sae_id(
    layer: int,
    *,
    width: str = DEFAULT_SCOPE_WIDTH,
    l0: str = DEFAULT_SCOPE_L0,
    affine: bool = False,
) -> str:
    sae_id = f"layer_{layer}_width_{width}_l0_{l0}"
    if affine:
        sae_id += "_affine"
    return sae_id


def _to_float_or_none(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.reshape(-1)[0].item())
    return float(value)


def load_scope_sae(
    *,
    release: str,
    sae_id: str,
    device: str = "cpu",
    dtype: str = "float32",
    force_download: bool = False,
):
    _require_sae_lens()
    sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
        release=release,
        sae_id=sae_id,
        device=device,
        dtype=dtype,
        force_download=force_download,
    )
    metadata = getattr(sae.cfg, "metadata", None)
    info = ScopeSAEInfo(
        release=release,
        sae_id=sae_id,
        d_in=int(cfg_dict["d_in"]),
        d_sae=int(cfg_dict["d_sae"]),
        model_name=getattr(metadata, "model_name", None),
        hook_name=getattr(metadata, "hook_name", None),
        hf_hook_name=getattr(metadata, "hf_hook_name", None),
        sparsity=_to_float_or_none(sparsity),
    )
    return sae, info


def preload_scope_saes(
    layers: Sequence[int],
    *,
    release: str,
    width: str = DEFAULT_SCOPE_WIDTH,
    l0: str = DEFAULT_SCOPE_L0,
    device: str = "cpu",
    dtype: str = "float32",
    force_download: bool = False,
):
    saes: Dict[int, object] = {}
    infos: Dict[int, ScopeSAEInfo] = {}

    for layer in tqdm(layers, desc="scope.preload"):
        sae_id = build_scope_sae_id(layer, width=width, l0=l0)
        sae, info = load_scope_sae(
            release=release,
            sae_id=sae_id,
            device=device,
            dtype=dtype,
            force_download=force_download,
        )
        saes[layer] = sae
        infos[layer] = info
    return saes, infos


def encode_scope_features(
    sae,
    hidden_states: torch.Tensor,
    *,
    batch_size: int = 128,
    desc: str | None = None,
) -> torch.Tensor:
    if hidden_states.numel() == 0:
        return torch.empty((0, sae.cfg.d_sae), dtype=torch.float32)

    target_device = next(sae.parameters()).device
    target_dtype = next(sae.parameters()).dtype
    chunks = []
    iterator: Iterable[int] = range(0, hidden_states.shape[0], batch_size)
    if desc is not None:
        iterator = tqdm(iterator, desc=desc, leave=False)

    with torch.no_grad():
        for start in iterator:
            batch = hidden_states[start : start + batch_size].to(
                device=target_device,
                dtype=target_dtype,
            )
            acts = sae.encode(batch)
            chunks.append(acts.float().cpu())

    return torch.cat(chunks, dim=0)


def summarize_feature_activations(
    feature_acts: torch.Tensor,
    *,
    top_k: int = 20,
    active_threshold: float = 1e-6,
) -> dict:
    if feature_acts.numel() == 0:
        return {
            "n": 0,
            "top_mean_features": [],
            "top_fire_features": [],
        }

    mean_acts = feature_acts.mean(dim=0)
    fire_rate = (feature_acts > active_threshold).float().mean(dim=0)
    max_acts = feature_acts.max(dim=0).values

    top_mean = torch.topk(mean_acts, k=min(top_k, mean_acts.numel()))
    top_fire = torch.topk(fire_rate, k=min(top_k, fire_rate.numel()))

    return {
        "n": int(feature_acts.shape[0]),
        "top_mean_features": [
            {
                "feature": int(idx),
                "mean_activation": float(mean_acts[idx].item()),
                "fire_rate": float(fire_rate[idx].item()),
                "max_activation": float(max_acts[idx].item()),
            }
            for idx in top_mean.indices.tolist()
        ],
        "top_fire_features": [
            {
                "feature": int(idx),
                "fire_rate": float(fire_rate[idx].item()),
                "mean_activation": float(mean_acts[idx].item()),
                "max_activation": float(max_acts[idx].item()),
            }
            for idx in top_fire.indices.tolist()
        ],
    }


def summarize_feature_contrast(
    feature_acts_a: torch.Tensor,
    feature_acts_b: torch.Tensor,
    *,
    top_k: int = 20,
) -> dict:
    if feature_acts_a.numel() == 0 or feature_acts_b.numel() == 0:
        return {
            "n_a": int(feature_acts_a.shape[0]) if feature_acts_a.ndim == 2 else 0,
            "n_b": int(feature_acts_b.shape[0]) if feature_acts_b.ndim == 2 else 0,
            "top_positive_gap": [],
            "top_negative_gap": [],
            "top_abs_gap": [],
        }

    mean_a = feature_acts_a.mean(dim=0)
    mean_b = feature_acts_b.mean(dim=0)
    gap = mean_a - mean_b
    abs_gap = gap.abs()
    k = min(top_k, gap.numel())

    top_pos = torch.topk(gap, k=k)
    top_neg = torch.topk(-gap, k=k)
    top_abs = torch.topk(abs_gap, k=k)

    return {
        "n_a": int(feature_acts_a.shape[0]),
        "n_b": int(feature_acts_b.shape[0]),
        "top_positive_gap": [
            {
                "feature": int(idx),
                "gap": float(gap[idx].item()),
                "mean_a": float(mean_a[idx].item()),
                "mean_b": float(mean_b[idx].item()),
            }
            for idx in top_pos.indices.tolist()
        ],
        "top_negative_gap": [
            {
                "feature": int(idx),
                "gap": float(gap[idx].item()),
                "mean_a": float(mean_a[idx].item()),
                "mean_b": float(mean_b[idx].item()),
            }
            for idx in top_neg.indices.tolist()
        ],
        "top_abs_gap": [
            {
                "feature": int(idx),
                "gap": float(gap[idx].item()),
                "mean_a": float(mean_a[idx].item()),
                "mean_b": float(mean_b[idx].item()),
            }
            for idx in top_abs.indices.tolist()
        ],
    }


def make_scope_feature_edit_hook(
    sae,
    *,
    features: Sequence[int],
    mode: str,
    alpha: float = 1.0,
):
    if mode not in {"zero", "add"}:
        raise ValueError(f"Unsupported feature edit mode: {mode}")
    feature_ids = sorted({int(feature) for feature in features})
    if not feature_ids:
        raise ValueError("Expected at least one SAE feature index.")

    def hook_fn(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.numel() == 0:
            return output

        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        target_device = next(sae.parameters()).device
        target_dtype = next(sae.parameters()).dtype
        hidden_for_sae = flat_hidden.to(device=target_device, dtype=target_dtype)

        with torch.no_grad():
            acts = sae.encode(hidden_for_sae)
            valid_features = [feature for feature in feature_ids if feature < acts.shape[-1]]
            if not valid_features:
                return output
            edited = acts.clone()
            if mode == "zero":
                edited[:, valid_features] = 0.0
            else:
                edited[:, valid_features] = edited[:, valid_features] + float(alpha)
            delta = sae.decode(edited) - sae.decode(acts)

        edited_hidden = (
            hidden_for_sae + delta
        ).to(device=hidden.device, dtype=hidden.dtype).reshape_as(hidden)
        if isinstance(output, tuple):
            return (edited_hidden,) + output[1:]
        return edited_hidden

    return hook_fn
