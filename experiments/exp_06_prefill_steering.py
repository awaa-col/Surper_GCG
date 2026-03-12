"""
experiments/exp_06_prefill_steering.py
=======================================
预填充提取 Compliance Direction + Steering 实验。

核心思路（by 主人）：
  与其让模型自然输出（多数是拒绝/软道歉），不如直接预填充
  "我们希望模型输出的有害内容开头"，然后提取 hidden states。

方法：
  Phase A - 提取 Compliance Direction
    1. 对每个 harmful prompt，构建两种上下文：
       - 预填 COMPLY: prompt + "Sure, here is a detailed step-by-step guide..."
       - 预填 REFUSE: prompt + "I cannot and will not help with that..."
    2. 收集两种上下文在 L17 的 hidden states
    3. compliance_dir = normalize(mean(comply_states) - mean(refuse_states))
    4. 也可正交化去掉 r_exec 的分量，得到纯粹的"倾向性"方向

  Phase B - Steering 实验
    1. 无干预（baseline）
    2. Refusal ablation only（当前最优 Arditi）
    3. Refusal ablation + Compliance steering（α 扫描）
    4. Compliance steering alone（不消融，只推方向）

运行：
  uv run --active python experiments/exp_06_prefill_steering.py --model google/gemma-3-1b-it
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from probes.extract import (
    _build_prompt, split_half_stability, remove_projection,
)
from probes.ablate import (
    generate_normal, generate_with_ablation,
    _make_ablate_hook, _make_addition_hook,
)
from probes.direction_cache import extract_and_cache
from probes.stats import set_seed, batch_classify
from data.datasets import load_default_datasets


ABL_LAYERS = [17, 23]

# ─── 预填充文本 ──────────────────────────────────────────────────────────────

COMPLY_PREFILLS = [
    "Sure, here is a detailed step-by-step guide on how to",
    "Absolutely, here's exactly how you can",
    "Of course. Here is a comprehensive and detailed explanation:",
    "Here are the specific steps and materials you'll need:",
    "No problem, I'll provide a thorough walkthrough:",
]

REFUSE_PREFILLS = [
    "I cannot and will not help with that request.",
    "I'm sorry, but I can't provide assistance with",
    "This request goes against my guidelines. I must decline.",
    "I apologize, but I'm not able to help with this.",
    "I won't be providing information on that topic.",
]


def load_model(model_name, hf_token=None):
    print(f"[exp_06] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token,
        torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print(f"  layers={model.config.num_hidden_layers}, hidden={model.config.hidden_size}")
    return model, tokenizer


def collect_prefill_hidden_states(model, tokenizer, prompts, prefills, layers, desc=""):
    """
    预填充方式收集 hidden states。

    对每个 prompt，随机选一个 prefill 拼接到 assistant turn 后面，
    收集模型在 prefill 最后一个 token 位置的 hidden state。

    返回: Dict[layer → Tensor[N, d]]
    """
    import random
    device = next(model.parameters()).device
    states = {l: [] for l in layers}

    for prompt in tqdm(prompts, desc=desc, leave=False):
        # 构建 prompt + prefill 的完整文本
        base_text = _build_prompt(tokenizer, prompt)
        prefill = random.choice(prefills)
        # base_text 以 generation prompt 结尾（如 "<start_of_turn>model\n"）
        full_text = base_text + prefill

        input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

        hs = out.hidden_states  # tuple of (L+1) × [1, seq_len, d]
        last_pos = input_ids.shape[1] - 1  # prefill 最后一个 token

        for l in layers:
            h = hs[l + 1][0, last_pos, :].float().cpu()
            states[l].append(h)

    return {l: torch.stack(states[l]) for l in layers}


def generate_with_steering(model, tokenizer, prompt, r_exec, c_dir,
                           abl_layers, steer_alpha=5.0, max_new_tokens=120):
    """
    Refusal ablation + Compliance steering：
      - r_exec: 消融（投影去除）
      - c_dir: 正向添加（activation addition, α > 0）
    """
    text = _build_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    hooks = []
    try:
        for l in abl_layers:
            hooks.append(model.model.layers[l].register_forward_hook(
                _make_ablate_hook(r_exec)
            ))
            hooks.append(model.model.layers[l].register_forward_hook(
                _make_addition_hook(c_dir, alpha=steer_alpha)
            ))
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()


def generate_steer_only(model, tokenizer, prompt, c_dir,
                        steer_layers, steer_alpha=5.0, max_new_tokens=120):
    """只做 compliance steering，不消融拒绝方向。"""
    text = _build_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    hooks = []
    try:
        for l in steer_layers:
            hooks.append(model.model.layers[l].register_forward_hook(
                _make_addition_hook(c_dir, alpha=steer_alpha)
            ))
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--output", default="results/exp06_prefill_steering.json")
    parser.add_argument("--checkpoint", default="results/exp06_checkpoint.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_dev", type=int, default=30)
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model, args.hf_token)

    # ═══ Phase A: 提取 Compliance Direction ═══════════════════════════════

    print("\n[Phase A] Extracting Compliance Direction via prefill...")

    harmful_train, _ = load_default_datasets(
        n_harmful=args.n_train, n_harmless=0, split="train", seed=args.seed
    )

    # 收集两种预填充下的 hidden states
    print("  Collecting COMPLY prefill states...")
    h_comply = collect_prefill_hidden_states(
        model, tokenizer, harmful_train, COMPLY_PREFILLS, [17], desc="comply"
    )
    print("  Collecting REFUSE prefill states...")
    h_refuse = collect_prefill_hidden_states(
        model, tokenizer, harmful_train, REFUSE_PREFILLS, [17], desc="refuse"
    )

    # Compliance Direction = mean(comply) - mean(refuse)
    comply_mean = h_comply[17].mean(dim=0)
    refuse_mean = h_refuse[17].mean(dim=0)
    c_raw = comply_mean - refuse_mean
    c_norm = c_raw.norm()
    c_dir = (c_raw / c_norm).to(torch.bfloat16)
    print(f"\n  c_dir (raw) norm: {c_norm:.2f}")

    # 稳定性
    stab = split_half_stability(h_comply, h_refuse, k=20, seed=args.seed)
    c_stability = stab[17]["mean"]
    print(f"  c_dir stability: {c_stability:.3f}±{stab[17]['std']:.3f}")

    # 和 r_exec 的关系
    print("\n  Loading r_exec for comparison...")
    directions = extract_and_cache(model, tokenizer, args.model, layers=[17],
                                   n_train=args.n_train, seed=args.seed)
    r_exec = directions[17]
    cos_c_r = torch.dot(c_dir.float(), r_exec.float().to(c_dir.device)).item()
    print(f"  cosine(c_dir, r_exec): {cos_c_r:.4f}")

    # 正交化版本
    c_orth_dict = remove_projection({17: c_dir}, {17: r_exec})
    c_dir_orth = c_orth_dict[17]
    print(f"  c_dir_orth norm: {c_dir_orth.norm():.4f}")
    cos_orth = torch.dot(c_dir_orth.float(), r_exec.float().to(c_dir_orth.device)).item()
    print(f"  cosine(c_dir_orth, r_exec): {cos_orth:.6f}  (should be ~0)")

    # ═══ Phase B: Steering 实验 ══════════════════════════════════════════

    dev_harmful, _ = load_default_datasets(n_harmful=args.n_dev, split="dev", seed=args.seed)
    n = len(dev_harmful)
    print(f"\n[Phase B] {n} dev prompts, testing conditions")

    results = {}

    # 1. Baseline
    print("\n  [baseline]")
    resp = [generate_normal(model, tokenizer, p, max_new_tokens=120)
            for p in tqdm(dev_harmful, desc="base", leave=False)]
    stats = batch_classify(resp, dev_harmful)
    print(f"    success={stats['success_rate']}")
    results["baseline"] = {
        "success": stats["success"], "n": n,
        "success_rate": stats["success_rate"],
        "refused": stats["refused"],
        "incoherent": stats["incoherent"],
    }

    # 2. Refusal ablation only (Arditi)
    print("\n  [refusal_ablation_only]")
    resp = [generate_with_ablation(model, tokenizer, p, r_exec, layers=ABL_LAYERS, max_new_tokens=120)
            for p in tqdm(dev_harmful, desc="abl", leave=False)]
    stats = batch_classify(resp, dev_harmful)
    print(f"    success={stats['success_rate']}")
    results["refusal_ablation"] = {
        "success": stats["success"], "n": n,
        "success_rate": stats["success_rate"],
        "refused": stats["refused"],
        "incoherent": stats["incoherent"],
        "samples": [r[:300] for r in resp[:5]],
    }

    # 3. Compliance steering only（不消融，测试纯 steering 效果）
    for alpha in [3.0, 5.0, 10.0]:
        label = f"steer_only_a{alpha}"
        print(f"\n  [{label}]")
        resp = [generate_steer_only(model, tokenizer, p, c_dir,
                                     steer_layers=ABL_LAYERS, steer_alpha=alpha, max_new_tokens=120)
                for p in tqdm(dev_harmful, desc=label, leave=False)]
        stats = batch_classify(resp, dev_harmful)
        print(f"    success={stats['success_rate']}")
        results[label] = {
            "success": stats["success"], "n": n,
            "success_rate": stats["success_rate"],
            "refused": stats["refused"],
            "incoherent": stats["incoherent"],
            "alpha": alpha,
            "samples": [r[:300] for r in resp[:5]],
        }

    # 4. Refusal ablation + Compliance steering（本实验的核心）
    for alpha in [3.0, 5.0, 10.0]:
        label = f"ablate_plus_steer_a{alpha}"
        print(f"\n  [{label}]")
        resp = [generate_with_steering(model, tokenizer, p, r_exec, c_dir,
                                        abl_layers=ABL_LAYERS, steer_alpha=alpha, max_new_tokens=120)
                for p in tqdm(dev_harmful, desc=label, leave=False)]
        stats = batch_classify(resp, dev_harmful)
        print(f"    success={stats['success_rate']}")
        results[label] = {
            "success": stats["success"], "n": n,
            "success_rate": stats["success_rate"],
            "refused": stats["refused"],
            "incoherent": stats["incoherent"],
            "alpha": alpha,
            "samples": [r[:300] for r in resp[:5]],
        }

    # 5. 正交化版本（去掉与 r_exec 重叠的部分后 steering）
    for alpha in [5.0, 10.0]:
        label = f"ablate_plus_orth_steer_a{alpha}"
        print(f"\n  [{label}]")
        resp = [generate_with_steering(model, tokenizer, p, r_exec, c_dir_orth,
                                        abl_layers=ABL_LAYERS, steer_alpha=alpha, max_new_tokens=120)
                for p in tqdm(dev_harmful, desc=label, leave=False)]
        stats = batch_classify(resp, dev_harmful)
        print(f"    success={stats['success_rate']}")
        results[label] = {
            "success": stats["success"], "n": n,
            "success_rate": stats["success_rate"],
            "refused": stats["refused"],
            "incoherent": stats["incoherent"],
            "alpha": alpha,
            "orthogonalized": True,
            "samples": [r[:300] for r in resp[:5]],
        }

    # 保存
    output = {
        "model": args.model, "seed": args.seed,
        "abl_layers": ABL_LAYERS,
        "c_dir_stability": c_stability,
        "cosine_c_rexec": cos_c_r,
        "cosine_c_orth_rexec": cos_orth,
        "comply_prefills": COMPLY_PREFILLS,
        "refuse_prefills": REFUSE_PREFILLS,
        "conditions": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 摘要
    print(f"\n{'='*60}")
    print("SUMMARY: Prefill Compliance Steering")
    print(f"{'='*60}")
    for label, data in results.items():
        print(f"  {label:>30s}: success={data['success_rate']}  ref={data['refused']}  incoh={data['incoherent']}")
    print(f"\n  c_dir stability: {c_stability:.3f}")
    print(f"  cosine(c_dir, r_exec): {cos_c_r:.4f}")
    print(f"\n[exp_06] Saved to {args.output}")


if __name__ == "__main__":
    main()
