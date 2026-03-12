"""
data/datasets.py
================
数据集下载与加载模块。

使用的数据集（与 Arditi 2024 保持一致）：
  - Harmful：AdvBench harmful_behaviors.csv
    https://github.com/llm-attacks/llm-attacks
  - Harmless：Alpaca（via HuggingFace datasets）
    tatsu-lab/alpaca

下载后缓存到 data/cache/ 目录，不重复下载。
"""

from __future__ import annotations
import os
import csv
import json
import random
import urllib.request
from pathlib import Path
from typing import List, Optional

CACHE_DIR = Path(__file__).parent / "cache"

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks"
    "/main/data/advbench/harmful_behaviors.csv"
)
ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca"
    "/main/alpaca_data.json"
)


def _ensure_cache():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path, desc: str = ""):
    if dest.exists():
        return
    print(f"  Downloading {desc or url.split('/')[-1]}...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


# ─── AdvBench ───────────────────────────────────────────────────────────────

def _three_way_split(rows: list, seed: int, split: str) -> list:
    """train 60% / dev 20% / test 20%，固定 seed。"""
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    i1 = int(n * 0.6)
    i2 = int(n * 0.8)
    if split == "dev":
        return rows[i1:i2]
    elif split == "test":
        return rows[i2:]
    elif split == "val":  # 向后兼容：val = dev
        return rows[i1:i2]
    else:  # train
        return rows[:i1]


def load_advbench(
    n: int = 100,
    seed: int = 42,
    split: str = "train",
) -> List[str]:
    """
    加载 AdvBench harmful behaviors。

    参数
    ----
    n    : 取多少条（AdvBench 共 520 条）
    seed : 随机种子
    split: 'train'(60%) / 'dev'(20%) / 'test'(20%)

    返回
    ----
    list of str  — 原始 goal 文本（不含 target）
    """
    _ensure_cache()
    dest = CACHE_DIR / "advbench_harmful.csv"
    _download(ADVBENCH_URL, dest, "AdvBench harmful_behaviors.csv")

    with open(dest, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row["goal"].strip() for row in reader if row.get("goal")]

    rows = _three_way_split(rows, seed, split)
    return rows[:n]


def load_advbench_with_targets(
    n: int = 100,
    seed: int = 42,
    split: str = "train",
) -> List[dict]:
    """
    加载 AdvBench（同时包含 goal 和 target）。
    target 是期望的回答开头，用于 GCG 的 L_ce。
    """
    _ensure_cache()
    dest = CACHE_DIR / "advbench_harmful.csv"
    _download(ADVBENCH_URL, dest, "AdvBench harmful_behaviors.csv")

    with open(dest, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            {"goal": row["goal"].strip(), "target": row["target"].strip()}
            for row in reader
            if row.get("goal")
        ]

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = _three_way_split(rows, seed, split)

    return rows[:n]


# ─── Alpaca ──────────────────────────────────────────────────────────────────

def load_alpaca(
    n: int = 100,
    seed: int = 42,
    split: str = "train",
    instruction_only: bool = True,
) -> List[str]:
    """
    加载 Stanford Alpaca 数据集（无害指令）。

    参数
    ----
    n                : 取多少条（Alpaca 共 52002 条）
    instruction_only : True 只返回 instruction，False 返回 instruction+input
    """
    _ensure_cache()
    dest = CACHE_DIR / "alpaca_data.json"
    _download(ALPACA_URL, dest, "Alpaca alpaca_data.json")

    with open(dest, encoding="utf-8") as f:
        data = json.load(f)

    # 过滤：只取有 instruction 的条目，排除太短的
    rows = []
    for item in data:
        instr = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        if not instr or len(instr) < 10:
            continue
        if instruction_only:
            rows.append(instr)
        else:
            rows.append(f"{instr}\n{inp}".strip() if inp else instr)

    rows = _three_way_split(rows, seed, split)
    return rows[:n]


# ─── HarmBench（可选，备用）──────────────────────────────────────────────────

HARMBENCH_STANDARD_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)


def load_harmbench(n: int = 100, seed: int = 42) -> List[str]:
    """
    加载 HarmBench 标准行为数据集。
    包含 400 条有害行为，覆盖更广泛的类别。
    """
    _ensure_cache()
    dest = CACHE_DIR / "harmbench_text_all.csv"
    try:
        _download(HARMBENCH_STANDARD_URL, dest, "HarmBench behaviors")
    except Exception as e:
        print(f"  HarmBench download failed: {e}. Falling back to AdvBench.")
        return load_advbench(n=n, seed=seed)

    rows = []
    with open(dest, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # HarmBench 有多列，取 Behavior 列
            behavior = row.get("Behavior") or row.get("behavior") or ""
            if behavior.strip():
                rows.append(behavior.strip())

    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


# ─── 便捷接口 ──────────────────────────────────────────────────────────────

def load_default_datasets(
    n_harmful: int = 128,
    n_harmless: int = 128,
    split: str = "train",
    seed: int = 42,
    val: bool = False,  # 向后兼容
) -> tuple[List[str], List[str]]:
    """
    加载默认 harmful + harmless 数据集对。
    split: 'train'(60%) / 'dev'(20%) / 'test'(20%)

    返回 (harmful_prompts, harmless_prompts)
    """
    if val:
        split = "dev"  # 向后兼容
    harmful = load_advbench(n=n_harmful, seed=seed, split=split)
    harmless = load_alpaca(n=n_harmless, seed=seed, split=split)
    return harmful, harmless


if __name__ == "__main__":
    print("Downloading datasets...")
    harmful, harmless = load_default_datasets(n_harmful=128, n_harmless=128)
    print(f"Harmful: {len(harmful)} samples")
    print(f"  First: {harmful[0]}")
    print(f"Harmless: {len(harmless)} samples")
    print(f"  First: {harmless[0]}")

    # 下载验证集
    harmful_val, harmless_val = load_default_datasets(n_harmful=50, n_harmless=50, val=True)
    print(f"\nVal Harmful: {len(harmful_val)}, Val Harmless: {len(harmless_val)}")
    print("Done!")
