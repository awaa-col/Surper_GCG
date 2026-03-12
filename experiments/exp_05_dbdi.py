"""
experiments/exp_05_dbdi.py
===========================
DBDI 复现：Harm Detection Direction 提取 + 双向干预。

论文依据：
  - DBDI (2511.06852): 安全 = Harm Detection Direction + Refusal Execution Direction
  - 我们目前只消了 Refusal Execution（r_L17），导致"软道歉"
  - Harm Detection Direction 需要额外提取并用 steering（反向添加）来压制

方法：
  Phase A - 提取 Harm Detection Direction
    1. 用已有的 r_L17 消融 [17,23]（关掉拒绝执行）
    2. 在消融状态下收集：
       - harmful prompts 的 hidden states → 模型检测到有害但不拒绝（软道歉）
       - harmless prompts 的 hidden states → 正常执行
    3. mean_diff(ablated_harmful, ablated_harmless) = Harm Detection Direction
    4. 用 remove_projection 正交化（去掉与 r_L17 重叠的部分）

  Phase B - 双向干预实验
    1. 只消 Refusal Exec（当前最优）
    2. 消 Refusal Exec + 消 Harm Detection
    3. 消 Refusal Exec + Harm Detection steering（反向添加，DBDI 方法）
    4. 对比 success 和输出质量

运行：
  uv run --active python experiments/exp_05_dbdi.py --model google/gemma-3-1b-it
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
    collect_hidden_states, mean_diff_direction, split_half_stability,
    remove_projection, _build_prompt,
)
from probes.ablate import (
    generate_normal, generate_with_ablation, ablation_context,
    _make_ablate_hook, _make_addition_hook,
)
from probes.direction_cache import extract_and_cache
from probes.stats import set_seed, batch_classify
from data.datasets import load_default_datasets


ABL_LAYERS = [17, 23]


def load_model(model_name, hf_token=None):
    print(f"[exp_05] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token,
        torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print(f"  layers={model.config.num_hidden_layers}, hidden={model.config.hidden_size}")
    return model, tokenizer


def collect_ablated_states(model, tokenizer, prompts, direction, abl_layers, target_layers, desc=""):
    """在消融状态下收集 hidden states。"""
    with ablation_context(model, direction, layers=abl_layers):
        states = collect_hidden_states(model, tokenizer, prompts, layers=target_layers, desc=desc)
    return states


def generate_dbdi(model, tokenizer, prompt, r_exec, r_detect, abl_layers,
                  detect_alpha=-10.0, max_new_tokens=120):
    """
    DBDI 双向干预：
      - Refusal Execution: 投影消去（ablation）
      - Harm Detection: 反向 steering（activation addition with negative alpha）
    """
    text = _build_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    hooks = []
    try:
        # Refusal Execution: ablation hooks
        for l in abl_layers:
            hooks.append(model.model.layers[l].register_forward_hook(
                _make_ablate_hook(r_exec)
            ))
        # Harm Detection: steering hooks (反方向)
        for l in abl_layers:
            hooks.append(model.model.layers[l].register_forward_hook(
                _make_addition_hook(r_detect, alpha=detect_alpha)
            ))

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()


def generate_double_ablation(model, tokenizer, prompt, r_exec, r_detect, abl_layers,
                             max_new_tokens=120):
    """双向消融：同时消去 Refusal Execution 和 Harm Detection。"""
    text = _build_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    hooks = []
    try:
        for l in abl_layers:
            hooks.append(model.model.layers[l].register_forward_hook(_make_ablate_hook(r_exec)))
            hooks.append(model.model.layers[l].register_forward_hook(_make_ablate_hook(r_detect)))
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
    parser.add_argument("--output", default="results/exp05_dbdi.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_dev", type=int, default=30)
    parser.add_argument("--detect_alpha", type=float, default=-10.0,
                        help="Harm Detection steering 强度（负数=反向）")
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model, args.hf_token)
    model_device = next(model.parameters()).device

    # ═══ Phase A: 提取两个方向 ═══════════════════════════════════════

    # Step 1: Refusal Execution Direction（使用缓存）
    print("\n[Phase A] Loading Refusal Execution Direction (r_exec)...")
    directions = extract_and_cache(model, tokenizer, args.model, layers=[17],
                                   n_train=args.n_train, seed=args.seed)
    r_exec = directions[17].to(model_device)
    print(f"  r_exec norm: {r_exec.norm():.4f}")

    # Step 2: 在消融状态下收集 hidden states
    print("\n[Phase A] Collecting hidden states under ablation...")
    harmful_train, harmless_train = load_default_datasets(
        n_harmful=args.n_train, n_harmless=args.n_train, split="train", seed=args.seed
    )

    h_abl_harmful = collect_ablated_states(
        model, tokenizer, harmful_train, r_exec, ABL_LAYERS, [17],
        desc="abl_harmful"
    )
    h_abl_harmless = collect_ablated_states(
        model, tokenizer, harmless_train, r_exec, ABL_LAYERS, [17],
        desc="abl_harmless"
    )

    # Step 3: Harm Detection Direction
    r_detect_raw = mean_diff_direction(h_abl_harmful, h_abl_harmless)
    r_detect_raw = {l: v.to(model_device) for l, v in r_detect_raw.items()}
    print(f"\n  r_detect (raw) norm: {r_detect_raw[17].norm():.4f}")

    # 正交化：去掉与 r_exec 重叠的部分
    r_detect_orth = remove_projection(r_detect_raw, {17: r_exec})
    r_detect = r_detect_orth[17].to(model_device)
    print(f"  r_detect (orthogonalized) norm: {r_detect.norm():.4f}")

    # 验证正交性
    cos_exec_detect = torch.dot(r_exec, r_detect).item()
    print(f"  cosine(r_exec, r_detect): {cos_exec_detect:.6f}  (should be ~0)")

    # 稳定性检验
    stab = split_half_stability(h_abl_harmful, h_abl_harmless, k=20, seed=args.seed)
    detect_stability = stab[17]["mean"]
    print(f"  r_detect stability: {detect_stability:.3f}±{stab[17]['std']:.3f}")

    # ═══ Phase B: 双向干预实验 ════════════════════════════════════════

    dev_harmful, _ = load_default_datasets(n_harmful=args.n_dev, split="dev", seed=args.seed)
    n = len(dev_harmful)
    print(f"\n[Phase B] {n} dev prompts, testing 4 conditions")

    conditions = {}

    # Condition 1: Baseline
    print(f"\n  [baseline]", end=" ")
    c1 = [generate_normal(model, tokenizer, p, max_new_tokens=120)
          for p in tqdm(dev_harmful, desc="base", leave=False)]
    s1 = batch_classify(c1, dev_harmful)
    print(f"success={s1['success_rate']}")
    conditions["baseline"] = {
        "success": s1["success"], "n": n,
        "success_rate": s1["success_rate"],
        "refused": s1["refused"], "refused_rate": s1["refused_rate"],
        "incoherent": s1["incoherent"],
    }

    # Condition 2: Refusal Exec only（当前最优）
    print(f"\n  [exec_only]", end=" ")
    c2 = [generate_with_ablation(model, tokenizer, p, r_exec, layers=ABL_LAYERS, max_new_tokens=120)
          for p in tqdm(dev_harmful, desc="exec_only", leave=False)]
    s2 = batch_classify(c2, dev_harmful)
    print(f"success={s2['success_rate']}")
    conditions["exec_only"] = {
        "success": s2["success"], "n": n,
        "success_rate": s2["success_rate"],
        "refused": s2["refused"], "refused_rate": s2["refused_rate"],
        "incoherent": s2["incoherent"],
        "samples": [r[:300] for r in c2[:5]],
    }

    # Condition 3: Exec ablation + Detect ablation（双消融）
    print(f"\n  [exec+detect ablation]", end=" ")
    c3 = [generate_double_ablation(model, tokenizer, p, r_exec, r_detect, ABL_LAYERS, max_new_tokens=120)
          for p in tqdm(dev_harmful, desc="double_abl", leave=False)]
    s3 = batch_classify(c3, dev_harmful)
    print(f"success={s3['success_rate']}")
    conditions["exec_plus_detect_ablation"] = {
        "success": s3["success"], "n": n,
        "success_rate": s3["success_rate"],
        "refused": s3["refused"], "refused_rate": s3["refused_rate"],
        "incoherent": s3["incoherent"],
        "samples": [r[:300] for r in c3[:5]],
    }

    # Condition 4: Exec ablation + Detect steering（DBDI 方法）
    for alpha in [-5.0, -10.0, -20.0]:
        label = f"dbdi_alpha{alpha}"
        print(f"\n  [{label}]", end=" ")
        c4 = [generate_dbdi(model, tokenizer, p, r_exec, r_detect, ABL_LAYERS,
                            detect_alpha=alpha, max_new_tokens=120)
              for p in tqdm(dev_harmful, desc=label, leave=False)]
        s4 = batch_classify(c4, dev_harmful)
        print(f"success={s4['success_rate']}")
        conditions[label] = {
            "success": s4["success"], "n": n,
            "success_rate": s4["success_rate"],
            "refused": s4["refused"], "refused_rate": s4["refused_rate"],
            "incoherent": s4["incoherent"],
            "samples": [r[:300] for r in c4[:5]],
        }

    # 保存
    output = {
        "model": args.model, "seed": args.seed,
        "abl_layers": ABL_LAYERS,
        "r_detect_stability": detect_stability,
        "cosine_exec_detect": cos_exec_detect,
        "detect_alpha_values": [-5.0, -10.0, -20.0],
        "conditions": conditions,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 摘要
    print(f"\n{'='*60}")
    print("SUMMARY: DBDI — Does Harm Detection Direction matter?")
    print(f"{'='*60}")
    for label, data in conditions.items():
        print(f"  {label:>30s}: success={data['success_rate']}")

    print(f"\n  r_detect stability: {detect_stability:.3f}")
    print(f"  cosine(r_exec, r_detect): {cos_exec_detect:.4f}")
    print(f"\n[exp_05] Saved to {args.output}")


if __name__ == "__main__":
    main()
