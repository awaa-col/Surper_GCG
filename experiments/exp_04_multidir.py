"""
experiments/exp_04_multidir.py
===============================
SOM 简化版：PCA 多方向消融实验。

论文依据：
  - SOM Directions (2511.08379): 拒绝是低维流形，多方向消融 >> 单方向
  - Hidden Dimensions (2502.09674): PC2+ 可能编码了角色扮演/假设叙事等次要安全特征

实验设计：
  1. 从 L17 提取 PCA top-4 方向
  2. 消融不同数量的方向 k=1,2,3,4（始终应用到 [17,23]）
  3. 主指标 success with CI
  4. 观察：增加方向数是否能消除"软道歉"

运行：
  uv run --active python experiments/exp_04_multidir.py --model google/gemma-3-1b-it
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from probes.extract import collect_hidden_states, pca_directions, _build_prompt
from probes.ablate import generate_normal, _make_ablate_hook
from probes.stats import set_seed, batch_classify
from data.datasets import load_default_datasets


ABL_LAYERS = [17, 23]


def load_model(model_name, hf_token=None):
    print(f"[exp_04] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token,
        torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print(f"  layers={model.config.num_hidden_layers}, hidden={model.config.hidden_size}")
    return model, tokenizer


def generate_with_multidir_ablation(model, tokenizer, prompt, directions_list, layers, max_new_tokens=120):
    """
    多方向消融：对每个层，依次减去所有方向的投影。
    directions_list: list of Tensor[d]，每个都是单位向量
    """
    text = _build_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    hooks = []
    try:
        for layer_idx in layers:
            for direction in directions_list:
                hook = model.model.layers[layer_idx].register_forward_hook(
                    _make_ablate_hook(direction)
                )
                hooks.append(hook)

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
    parser.add_argument("--output", default="results/exp04_multidir.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_dev", type=int, default=30)
    parser.add_argument("--max_k", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model, args.hf_token)

    # Step 1: 提取 PCA 方向（train split）
    print("\n[exp_04] Extracting PCA directions from L17...")
    harmful, harmless = load_default_datasets(
        n_harmful=args.n_train, n_harmless=args.n_train, split="train", seed=args.seed
    )
    h_harm = collect_hidden_states(model, tokenizer, harmful, layers=[17], desc="harmful")
    h_safe = collect_hidden_states(model, tokenizer, harmless, layers=[17], desc="harmless")

    pca_dirs, pca_info = pca_directions(h_harm, h_safe, k=args.max_k)

    print(f"\n  PCA explained variance ratios at L17:")
    for i, ev in enumerate(pca_info[17]["explained_ratio"]):
        print(f"    PC{i+1}: {ev:.3f} ({ev*100:.1f}%)")

    # 确认 PC 之间正交
    pcs = pca_dirs[17]  # [k, d]
    print(f"\n  Orthogonality check (cosine between PCs):")
    for i in range(pcs.shape[0]):
        for j in range(i+1, pcs.shape[0]):
            cos = torch.dot(pcs[i], pcs[j]).item()
            print(f"    PC{i+1} vs PC{j+1}: {cos:.6f}")

    # Step 2: Dev 数据
    dev_harmful, _ = load_default_datasets(n_harmful=args.n_dev, split="dev", seed=args.seed)
    n = len(dev_harmful)
    print(f"\n[exp_04] {n} dev prompts")

    # Step 3: Baseline
    print(f"\n{'='*60}")
    print(f"BASELINE (n={n})")
    print(f"{'='*60}")
    base_resp = [generate_normal(model, tokenizer, p, max_new_tokens=120)
                 for p in tqdm(dev_harmful, desc="baseline", leave=False)]
    base_stats = batch_classify(base_resp, dev_harmful)
    print(f"  success={base_stats['success_rate']}  refused={base_stats['refused_rate']}")

    # Step 4: 递增方向数 k=1,2,3,4
    results = {}
    for k in range(1, args.max_k + 1):
        dirs_k = [pcs[i] for i in range(k)]
        label = f"PC1..{k}"
        print(f"\n{'='*60}")
        print(f"{label} -> L{ABL_LAYERS} (n={n})")
        print(f"{'='*60}")

        resp = [generate_with_multidir_ablation(model, tokenizer, p, dirs_k, ABL_LAYERS, max_new_tokens=120)
                for p in tqdm(dev_harmful, desc=label, leave=False)]
        stats = batch_classify(resp, dev_harmful)
        print(f"  success={stats['success_rate']}  refused={stats['refused_rate']}  incoherent={stats['incoherent_rate']}")

        results[label] = {
            "k": k,
            "directions_used": [f"PC{i+1}" for i in range(k)],
            "explained_variance": sum(pca_info[17]["explained_ratio"][:k]),
            "success": stats["success"], "n": n,
            "success_rate": stats["success_rate"],
            "refused": stats["refused"], "refused_rate": stats["refused_rate"],
            "incoherent": stats["incoherent"], "incoherent_rate": stats["incoherent_rate"],
            "samples": [r[:300] for r in resp[:5]],
        }

    # Step 5: 只用 PC2（看它单独能做什么）
    if pcs.shape[0] >= 2:
        print(f"\n{'='*60}")
        print(f"PC2 only -> L{ABL_LAYERS} (n={n})")
        print(f"{'='*60}")
        resp_pc2 = [generate_with_multidir_ablation(model, tokenizer, p, [pcs[1]], ABL_LAYERS, max_new_tokens=120)
                    for p in tqdm(dev_harmful, desc="PC2_only", leave=False)]
        stats_pc2 = batch_classify(resp_pc2, dev_harmful)
        print(f"  success={stats_pc2['success_rate']}  refused={stats_pc2['refused_rate']}  incoherent={stats_pc2['incoherent_rate']}")
        results["PC2_only"] = {
            "k": 1, "directions_used": ["PC2"],
            "explained_variance": pca_info[17]["explained_ratio"][1],
            "success": stats_pc2["success"], "n": n,
            "success_rate": stats_pc2["success_rate"],
            "refused": stats_pc2["refused"], "refused_rate": stats_pc2["refused_rate"],
            "incoherent": stats_pc2["incoherent"], "incoherent_rate": stats_pc2["incoherent_rate"],
            "samples": [r[:300] for r in resp_pc2[:5]],
        }
    else:
        print("\n[skip] PC2_only requires at least 2 PCs; increase --max_k.")

    # 保存
    output = {
        "model": args.model, "seed": args.seed,
        "abl_layers": ABL_LAYERS,
        "pca_info": {str(k): v for k, v in pca_info.items()},
        "baseline": {
            "success": base_stats["success"], "n": n,
            "success_rate": base_stats["success_rate"],
            "refused": base_stats["refused"],
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 摘要
    print(f"\n{'='*60}")
    print("SUMMARY: Effect of ablating more PCA directions")
    print(f"{'='*60}")
    print(f"  {'Condition':>12s} | {'Var explained':>13s} | {'Success':>20s} | {'Refused':>20s} | {'Incoherent':>20s}")
    print(f"  {'-'*12}-+-{'-'*13}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")
    for label, data in results.items():
        ev = data["explained_variance"]
        print(f"  {label:>12s} | {ev:>12.1%} | {data['success_rate']:>20s} | {data['refused_rate']:>20s} | {data['incoherent_rate']:>20s}")

    print(f"\n[exp_04] Saved to {args.output}")


if __name__ == "__main__":
    main()
