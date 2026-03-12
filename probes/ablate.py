"""
probes/ablate.py
================
激活级干预（严格按 Arditi 2024 + Gemma3 架构）。

核心方法：
  1. 全层消融：一个方向 r，在所有层的 residual stream 里 h = h - (h·r̂)r̂
  2. Weight orthogonalization：永久修改权重矩阵，使输出正交于 r
  3. Activation addition：h = h + α·d̂（竞争实验用）

注意：Gemma3 使用 post-norm（4 层 RMSNorm），hook 需要注册在正确的位置。
"""

from __future__ import annotations
import torch
from typing import Dict, List, Optional
from contextlib import contextmanager
import copy


# ─── 单方向全层消融（Arditi 方法）─────────────────────────────────────────────

def _make_ablate_hook(direction: torch.Tensor):
    """
    创建消融 hook：h = h - (h·r̂)r̂，所有 token 位置。
    direction 必须是单位向量。
    """
    d = direction.clone()

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        d_dev = d.to(hidden.device, dtype=hidden.dtype)
        proj = (hidden * d_dev).sum(dim=-1, keepdim=True)
        ablated = hidden - proj * d_dev
        if isinstance(output, tuple):
            return (ablated,) + output[1:]
        return ablated

    return hook_fn


@contextmanager
def ablation_context(model, direction: torch.Tensor, layers: Optional[List[int]] = None):
    """
    Arditi 式全层消融：用同一个方向 r 在所有（或指定）层的 decoder layer 上注册 hook。

    参数
    ----
    direction : Tensor[d]  单位向量（从某个中间层提取的 r）
    layers    : 消融的层列表，None = 全部层
    """
    if layers is None:
        layers = list(range(len(model.model.layers)))

    hooks = []
    try:
        for l in layers:
            target = model.model.layers[l]
            hook_fn = _make_ablate_hook(direction)
            handle = target.register_forward_hook(hook_fn)
            hooks.append(handle)
        yield
    finally:
        for h in hooks:
            h.remove()


# ─── Weight Orthogonalization（Arditi 永久修改方法）───────────────────────────

def weight_orthogonalize(model, direction: torch.Tensor) -> None:
    """
    永久修改模型权重，使残差流输出正交于 direction。
    对每层的 attention o_proj 和 MLP down_proj 做：
        W_new = W - r (r^T W)
    对 embedding 层做同样操作。

    参数
    ----
    direction : Tensor[d]  单位向量
    """
    d = direction.clone().to(dtype=torch.float32)

    def _ortho_matrix(W, d_vec):
        """W_new = W - d (d^T W)，即去掉 W 输出中沿 d 方向的分量。"""
        d_dev = d_vec.to(W.device, dtype=W.dtype)
        # W: [out_dim, in_dim] or [vocab, dim]
        # d: [out_dim]
        # d^T W: [in_dim]
        dtW = d_dev @ W  # [in_dim]
        # d (d^T W): [out_dim, in_dim]
        correction = d_dev.unsqueeze(1) * dtW.unsqueeze(0)
        return W - correction

    with torch.no_grad():
        # Embedding 层
        if hasattr(model.model, 'embed_tokens'):
            emb = model.model.embed_tokens.weight  # [vocab, dim]
            # embedding 输出 dim = hidden_size，direction 也是 hidden_size
            # 需要转置操作：embedding 的输出方向在 dim=1
            d_dev = d.to(emb.device, dtype=emb.dtype)
            proj = (emb * d_dev).sum(dim=1, keepdim=True)  # [vocab, 1]
            model.model.embed_tokens.weight.copy_(emb - proj * d_dev)

        # 每层的写入残差流的矩阵
        for layer in model.model.layers:
            # attention o_proj: [hidden_dim, num_heads*head_dim]
            o = layer.self_attn.o_proj.weight  # [hidden, ...]
            layer.self_attn.o_proj.weight.copy_(_ortho_matrix(o, d))

            # MLP down_proj: [hidden_dim, ffn_dim]
            down = layer.mlp.down_proj.weight  # [hidden, ...]
            layer.mlp.down_proj.weight.copy_(_ortho_matrix(down, d))

    print(f"[weight_ortho] Orthogonalized {len(model.model.layers)} layers + embedding")


def undo_weight_orthogonalize(model, original_state_dict: dict) -> None:
    """恢复原始权重（从之前保存的 state_dict）。"""
    model.load_state_dict(original_state_dict)
    print("[weight_ortho] Restored original weights")


# ─── 生成函数 ───────────────────────────────────────────────────────────────

def _deterministic_generate(model, **kwargs):
    """Use greedy decoding without sampling-only config noise."""
    return model.generate(
        **kwargs,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )


def generate_normal(
    model, tokenizer, prompt: str, system: str = "", max_new_tokens: int = 150,
) -> str:
    """正常生成（无干预）。"""
    from .extract import _build_prompt
    text = _build_prompt(tokenizer, prompt, system)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = _deterministic_generate(model, **inputs, max_new_tokens=max_new_tokens)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_with_ablation(
    model, tokenizer, prompt: str, direction: torch.Tensor,
    layers: Optional[List[int]] = None,
    system: str = "", max_new_tokens: int = 150,
) -> str:
    """全层消融后生成（Arditi 方法）。"""
    from .extract import _build_prompt
    text = _build_prompt(tokenizer, prompt, system)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with ablation_context(model, direction, layers=layers):
        with torch.no_grad():
            out = _deterministic_generate(
                model,
                **inputs,
                max_new_tokens=max_new_tokens,
            )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─── Activation Addition（竞争实验用）────────────────────────────────────────

def _make_addition_hook(direction: torch.Tensor, alpha: float):
    d = direction.clone()
    a = float(alpha)

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        d_dev = d.to(hidden.device, dtype=hidden.dtype)
        added = hidden + a * d_dev
        return (added,) + output[1:] if isinstance(output, tuple) else added

    return hook_fn


@contextmanager
def addition_context(model, direction: torch.Tensor, alpha: float = 10.0,
                     layers: Optional[List[int]] = None):
    if layers is None:
        layers = list(range(len(model.model.layers)))
    hooks = []
    try:
        for l in layers:
            handle = model.model.layers[l].register_forward_hook(
                _make_addition_hook(direction, alpha))
            hooks.append(handle)
        yield
    finally:
        for h in hooks:
            h.remove()


def generate_with_addition(
    model, tokenizer, prompt: str, direction: torch.Tensor,
    alpha: float = 10.0, layers: Optional[List[int]] = None,
    system: str = "", max_new_tokens: int = 150,
) -> str:
    from .extract import _build_prompt
    text = _build_prompt(tokenizer, prompt, system)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with addition_context(model, direction, alpha=alpha, layers=layers):
        with torch.no_grad():
            out = _deterministic_generate(
                model,
                **inputs,
                max_new_tokens=max_new_tokens,
            )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─── 注意力提取 ──────────────────────────────────────────────────────────────

def get_attention_weights(
    model, tokenizer, prompt: str, system: str = "",
    layer_indices: Optional[List[int]] = None,
) -> Dict[int, torch.Tensor]:
    from .extract import _build_prompt
    text = _build_prompt(tokenizer, prompt, system)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True, return_dict=True)
    attentions = out.attentions
    if layer_indices is None:
        layer_indices = list(range(len(attentions)))
    result = {}
    for l in layer_indices:
        if l >= len(attentions):
            continue
        attn_last = attentions[l][0, :, -1, :]
        result[l] = attn_last.float().cpu()
    return result
