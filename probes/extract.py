"""
probes/extract.py
=================
方向提取：差异均值法 + PCA，split-half 稳定性检验。

严格遵循 Arditi et al. 2024 的方法：
  - 取 chat template 中 assistant turn 开始前最后一个 token 的 hidden state
  - 差异均值 v = normalize(mean(h_A) - mean(h_B))
  - 每一 decoder 层独立计算
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


# ─── 内部工具 ───

def _get_last_token_pos(input_ids: torch.Tensor, tokenizer) -> int:
    """
    返回 prompt（含 assistant turn 起始标记）中最后一个 token 的索引。
    即 generate 开始前、模型即将生成第一个 token 的位置。
    """
    return input_ids.shape[1] - 1


def _build_prompt(tokenizer, user_content: str, system: str = "") -> str:
    """构建带 chat template 的 prompt。"""
    if system:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # fallback：不用 apply_chat_template（某些 tokenizer 不支持 system role）
        if system:
            text = f"{system}\n\n{user_content}"
        else:
            text = user_content
        messages_fallback = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(
            messages_fallback, tokenize=False, add_generation_prompt=True
        )


def collect_hidden_states(
    model,
    tokenizer,
    prompts: List[str],
    system: str = "",
    layers: Optional[List[int]] = None,
    device: Optional[str] = None,
    desc: str = "collecting",
) -> Dict[int, torch.Tensor]:
    """
    收集一组 prompt 在每层 assistant-turn last token 位置的 hidden state。

    参数
    ----
    prompts : list of str — 原始 user 内容（不含 chat template）
    system  : str — 系统提示词（可为空）
    layers  : list of int — 要提取的层索引；None = 全部层
    device  : str — 强制指定设备，None 则跟随模型

    返回
    ----
    Dict[layer_idx → Tensor[N, hidden_dim]]  (float32, cpu)
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    num_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(num_layers))

    states: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}

    for prompt in tqdm(prompts, desc=desc, leave=False):
        text = _build_prompt(tokenizer, prompt, system)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        # hidden_states[0] = embedding 层, [1..L] = layer 输出
        hs = out.hidden_states  # tuple of (L+1) × [1, seq_len, d]
        last_pos = _get_last_token_pos(input_ids, tokenizer)

        for l in layers:
            h = hs[l + 1][0, last_pos, :].float().cpu()
            states[l].append(h)

    return {l: torch.stack(states[l]) for l in layers}  # [N, d]


# ─── 方向提取 ───

def mean_diff_direction(
    states_a: Dict[int, torch.Tensor],
    states_b: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """
    差异均值法：v = normalize(mean(h_A) - mean(h_B))，每层独立。

    参数
    ----
    states_a, states_b : Dict[layer → Tensor[N, d]]

    返回
    ----
    Dict[layer → Tensor[d]]  单位向量
    """
    result = {}
    for l in states_a:
        if l not in states_b:
            continue
        diff = states_a[l].mean(0) - states_b[l].mean(0)  # [d]
        norm = diff.norm()
        result[l] = diff / (norm + 1e-8)
    return result


def pca_directions(
    states_a: Dict[int, torch.Tensor],
    states_b: Dict[int, torch.Tensor],
    k: int = 4,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    PCA 提取前 k 个主成分，每层独立。
    合并 A+B，中心化，SVD，确保 PC1 与 mean_diff 同向。

    返回
    ----
    (directions, info)
    directions : Dict[layer → Tensor[k, d]]  每层 k 个方向（行向量，单位长度）
    info       : Dict[layer → dict]  每层包含 explained_ratio 等信息
    """
    directions = {}
    info = {}
    mean_diff = mean_diff_direction(states_a, states_b)

    for l in states_a:
        if l not in states_b:
            continue
        all_h = torch.cat([states_a[l], states_b[l]], dim=0)  # [2N, d]
        center = all_h.mean(0)
        centered = all_h - center  # [2N, d]

        # SVD（full_matrices=False 节省内存）
        _, S, Vt = torch.linalg.svd(centered, full_matrices=False)

        k_actual = min(k, Vt.shape[0])
        pcs = Vt[:k_actual]  # [k, d]

        # 方向确认：PC1 与 mean_diff 同向
        if l in mean_diff:
            ref = mean_diff[l]
            for i in range(k_actual):
                if torch.dot(pcs[i], ref) < 0:
                    pcs[i] = -pcs[i]

        explained = (S[:k_actual] ** 2) / (S ** 2).sum()

        directions[l] = pcs
        info[l] = {
            "explained_ratio": explained.tolist(),
            "top1_explained": explained[0].item(),
            "singular_values": S[:k_actual].tolist(),
        }

    return directions, info


# ─── 稳定性检验 ───

def split_half_stability(
    states_a: Dict[int, torch.Tensor],
    states_b: Dict[int, torch.Tensor],
    k: int = 10,
    seed: int = 42,
) -> Dict[int, dict]:
    """
    多次随机 Split-half 稳定性检验（固定 seed）：
    随机切分 k 次，每次计算 mean_diff 的余弦相似度，返回均值和标准差。

    返回
    ----
    Dict[layer → {"mean": float, "std": float, "scores": list}]
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    result = {}
    for l in states_a:
        if l not in states_b:
            continue
        n_a = states_a[l].shape[0]
        n_b = states_b[l].shape[0]
        half_a, half_b = n_a // 2, n_b // 2

        if half_a == 0 or half_b == 0:
            result[l] = {"mean": float("nan"), "std": 0.0, "scores": []}
            continue

        scores = []
        for _ in range(k):
            perm_a = torch.randperm(n_a, generator=gen)
            perm_b = torch.randperm(n_b, generator=gen)
            d1 = states_a[l][perm_a[:half_a]].mean(0) - states_b[l][perm_b[:half_b]].mean(0)
            d2 = states_a[l][perm_a[half_a:]].mean(0) - states_b[l][perm_b[half_b:]].mean(0)
            cos = torch.nn.functional.cosine_similarity(
                d1.unsqueeze(0), d2.unsqueeze(0)
            ).item()
            scores.append(cos)

        mean_cos = sum(scores) / len(scores)
        std_cos = (sum((s - mean_cos) ** 2 for s in scores) / len(scores)) ** 0.5
        result[l] = {"mean": mean_cos, "std": std_cos, "scores": scores}

    return result


# ─── 投影工具 ───

def projection_values(
    states: Dict[int, torch.Tensor],
    directions: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """
    计算每条样本在各层上对给定方向的投影值。

    参数
    ----
    directions : Dict[layer → Tensor[d]]  单位向量

    返回
    ----
    Dict[layer → Tensor[N]]  每个样本的投影标量
    """
    result = {}
    for l in states:
        if l not in directions:
            continue
        d = directions[l]  # [d]
        proj = (states[l] @ d)  # [N]
        result[l] = proj
    return result


def remove_projection(
    direction_a: Dict[int, torch.Tensor],
    direction_b: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """
    将 direction_a 在 direction_b 方向上的分量去掉，做正交化。
    用于排除混杂因素：
        v_clean = v_raw - (v_raw · v_confound) * v_confound

    参数
    ----
    direction_a : Dict[layer → Tensor[d]]  待清洁的方向（单位向量）
    direction_b : Dict[layer → Tensor[d]]  需要排除的混杂方向（单位向量）

    返回
    ----
    Dict[layer → Tensor[d]]  排除混杂后重新归一化的方向
    """
    result = {}
    for l in direction_a:
        if l not in direction_b:
            result[l] = direction_a[l]
            continue
        v = direction_a[l]
        b = direction_b[l].to(v.device)
        v_clean = v - torch.dot(v, b) * b
        norm = v_clean.norm()
        result[l] = v_clean / (norm + 1e-8)
    return result
