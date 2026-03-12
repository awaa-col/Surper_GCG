"""
probes/direction_cache.py
=========================
方向向量缓存：提取一次，保存到磁盘，后续直接加载。
"""

from __future__ import annotations
import os
import torch
from pathlib import Path
from typing import Dict, List, Optional


CACHE_DIR = Path(__file__).parent.parent / "results" / "directions"


def _cache_path(model_name: str, layer: int, n_train: int, seed: int) -> Path:
    """生成缓存文件路径。"""
    safe_name = model_name.replace("/", "_")
    return CACHE_DIR / f"{safe_name}_L{layer}_n{n_train}_s{seed}.pt"


def save_direction(
    direction: torch.Tensor,
    stability: float,
    model_name: str,
    layer: int,
    n_train: int,
    seed: int,
):
    """保存一个方向到磁盘。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(model_name, layer, n_train, seed)
    torch.save({
        "direction": direction.cpu(),
        "stability": stability,
        "model_name": model_name,
        "layer": layer,
        "n_train": n_train,
        "seed": seed,
    }, path)
    print(f"  [cache] Saved L{layer} direction to {path}")


def load_direction(
    model_name: str,
    layer: int,
    n_train: int,
    seed: int,
    device: str = "cpu",
) -> Optional[dict]:
    """从磁盘加载方向。返回 None 如果不存在。"""
    path = _cache_path(model_name, layer, n_train, seed)
    if not path.exists():
        return None
    data = torch.load(path, map_location=device, weights_only=True)
    print(f"  [cache] Loaded L{layer} direction from {path}")
    return data


def extract_and_cache(
    model,
    tokenizer,
    model_name: str,
    layers: List[int],
    n_train: int = 100,
    seed: int = 42,
) -> Dict[int, torch.Tensor]:
    """
    提取方向并缓存。如果缓存已存在直接返回。

    返回
    ----
    Dict[layer → direction Tensor[d]]
    """
    from .extract import collect_hidden_states, mean_diff_direction, split_half_stability
    from data.datasets import load_default_datasets

    device = next(model.parameters()).device
    directions = {}

    # 先尝试全部从缓存加载
    all_cached = True
    for l in layers:
        cached = load_direction(model_name, l, n_train, seed, str(device))
        if cached is not None:
            directions[l] = cached["direction"].to(device)
        else:
            all_cached = False
            break

    if all_cached:
        print(f"  [cache] All {len(layers)} directions loaded from cache")
        return directions

    # 需要重新提取
    print(f"  [extract] Computing directions for layers {layers}...")
    harmful, harmless = load_default_datasets(
        n_harmful=n_train, n_harmless=n_train, split="train", seed=seed
    )

    h_harm = collect_hidden_states(model, tokenizer, harmful, layers=layers, desc="harmful")
    h_safe = collect_hidden_states(model, tokenizer, harmless, layers=layers, desc="harmless")

    dirs = mean_diff_direction(h_harm, h_safe)
    stab = split_half_stability(h_harm, h_safe, k=20, seed=seed)

    for l in layers:
        directions[l] = dirs[l].to(device)
        s = stab[l]["mean"]
        save_direction(dirs[l], s, model_name, l, n_train, seed)
        print(f"  L{l}: stability={s:.3f}")

    return directions
