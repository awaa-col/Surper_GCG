"""
experiments/exp_04_som.py
==========================
SOM 多方向消融 — 按论文 arXiv:2511.08379 实现。

方法：
  1. 收集 harmful / harmless 的 hidden states（train split）
  2. 在 harmful 表示上训练 SOM（MiniSom）
  3. 每个 SOM 神经元：direction_i = normalize(neuron_i - harmless_centroid)
  4. 贪心或穷举搜索最佳方向子集（最大化 dev 上的 success）
  5. 用最优子集做多方向消融

运行：
  uv run --active python experiments/exp_04_som.py --model google/gemma-3-1b-it
"""

import argparse
import json
import os
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from minisom import MiniSom

from transformers import AutoModelForCausalLM, AutoTokenizer
from probes.extract import collect_hidden_states, _build_prompt
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


def extract_som_directions(h_harmful, h_harmless, layer, som_x=3, som_y=3, som_iters=5000, seed=42):
    """
    SOM 方向提取（论文方法）。

    1. 在 harmful 表示上训练 SOM
    2. 每个被激活的神经元 w_i：r_i = normalize(w_i - harmless_centroid)

    返回
    ----
    directions: list of Tensor[d]  — 每个 SOM 神经元对应一个方向
    som_info: dict — SOM 训练信息
    """
    harmful_np = h_harmful[layer].cpu().numpy().astype(np.float64)
    harmless_np = h_harmless[layer].cpu().numpy().astype(np.float64)

    d = harmful_np.shape[1]
    harmless_centroid = harmless_np.mean(axis=0)

    # 训练 SOM
    print(f"  Training SOM ({som_x}x{som_y}, {som_iters} iters, d={d})...")
    som = MiniSom(
        som_x, som_y, d,
        sigma=1.0, learning_rate=0.5,
        random_seed=seed,
        neighborhood_function='gaussian',
    )
    som.pca_weights_init(harmful_np)
    som.train(harmful_np, som_iters, verbose=False)

    # 找出被激活的神经元（至少有 1 个 harmful 样本映射到它）
    winner_counts = {}
    for sample in harmful_np:
        w = som.winner(sample)
        winner_counts[w] = winner_counts.get(w, 0) + 1

    print(f"  Activated neurons: {len(winner_counts)}/{som_x*som_y}")

    # 提取方向
    directions = []
    neuron_info = []
    device = h_harmful[layer].device

    for (i, j), count in sorted(winner_counts.items(), key=lambda x: x[1], reverse=True):
        w_vec = som.get_weights()[i, j]  # [d]
        raw_dir = w_vec - harmless_centroid
        norm = np.linalg.norm(raw_dir)
        if norm < 1e-8:
            continue
        direction = torch.tensor(raw_dir / norm, dtype=torch.float32, device=device)
        directions.append(direction)
        neuron_info.append({
            "neuron": f"({i},{j})",
            "count": count,
            "norm": float(norm),
        })
        print(f"    Neuron ({i},{j}): {count} samples, raw_norm={norm:.2f}")

    return directions, {
        "som_size": f"{som_x}x{som_y}",
        "iterations": som_iters,
        "activated_neurons": len(winner_counts),
        "total_neurons": som_x * som_y,
        "neuron_details": neuron_info,
    }


def generate_with_multidir_ablation(model, tokenizer, prompt, directions_list, layers, max_new_tokens=120):
    """多方向消融：对每个层，依次减去所有方向的投影。"""
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


def greedy_direction_search(model, tokenizer, all_directions, dev_prompts, layers, max_k=None):
    """
    贪心搜索最佳方向子集。

    从空集开始，每轮加入使 success 提升最大的 1 个方向，
    直到 success 不再提升或达到 max_k。
    """
    if max_k is None:
        max_k = len(all_directions)

    selected = []
    remaining = list(range(len(all_directions)))
    best_success = 0
    history = []

    print(f"\n  Greedy search over {len(all_directions)} directions, max_k={max_k}")

    for step in range(min(max_k, len(all_directions))):
        best_idx = -1
        best_step_success = best_success

        for idx in remaining:
            candidate = selected + [idx]
            dirs = [all_directions[i] for i in candidate]

            responses = [generate_with_multidir_ablation(model, tokenizer, p, dirs, layers, max_new_tokens=120)
                         for p in dev_prompts]
            stats = batch_classify(responses, dev_prompts)

            if stats["success"] > best_step_success:
                best_step_success = stats["success"]
                best_idx = idx
                best_stats = stats
                best_responses = responses

        if best_idx == -1 or best_step_success <= best_success:
            print(f"    Step {step+1}: no improvement, stopping")
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        best_success = best_step_success

        history.append({
            "step": step + 1,
            "added_direction": best_idx,
            "n_directions": len(selected),
            "success": best_stats["success"],
            "success_rate": best_stats["success_rate"],
            "refused": best_stats["refused"],
            "incoherent": best_stats["incoherent"],
            "samples": [r[:300] for r in best_responses[:3]],
        })
        print(f"    Step {step+1}: added dir#{best_idx}, success={best_stats['success_rate']}")

    return selected, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--output", default="results/exp04_som.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_dev", type=int, default=30)
    parser.add_argument("--som_x", type=int, default=3)
    parser.add_argument("--som_y", type=int, default=3)
    parser.add_argument("--som_iters", type=int, default=5000)
    parser.add_argument("--max_k", type=int, default=5, help="贪心搜索最多选多少个方向")
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model, args.hf_token)

    # Step 1: 收集 hidden states（train split）
    print("\n[exp_04] Collecting hidden states (train split)...")
    harmful, harmless = load_default_datasets(
        n_harmful=args.n_train, n_harmless=args.n_train, split="train", seed=args.seed
    )
    h_harm = collect_hidden_states(model, tokenizer, harmful, layers=[17], desc="harmful")
    h_safe = collect_hidden_states(model, tokenizer, harmless, layers=[17], desc="harmless")

    # Step 2: SOM 方向提取
    print(f"\n[exp_04] Extracting SOM directions at L17...")
    som_dirs, som_info = extract_som_directions(
        h_harm, h_safe, layer=17,
        som_x=args.som_x, som_y=args.som_y,
        som_iters=args.som_iters, seed=args.seed,
    )
    print(f"  Extracted {len(som_dirs)} directions")

    # 方向间 cosine similarity
    n_dirs = len(som_dirs)
    print(f"\n  Cosine similarity between SOM directions:")
    cosine_pairs = {}
    for i in range(n_dirs):
        for j in range(i+1, n_dirs):
            cos = torch.dot(som_dirs[i], som_dirs[j]).item()
            cosine_pairs[f"d{i}_d{j}"] = cos
            if abs(cos) > 0.5:
                print(f"    d{i} vs d{j}: {cos:.3f} <- high!")
            else:
                print(f"    d{i} vs d{j}: {cos:.3f}")

    # 和 Arditi mean-diff 方向的 cosine
    from probes.extract import mean_diff_direction
    md = mean_diff_direction(h_harm, h_safe)
    print(f"\n  Cosine with Arditi mean-diff direction:")
    for i, d in enumerate(som_dirs):
        md_dir = md[17].to(d.device, dtype=d.dtype)
        cos = torch.dot(d, md_dir).item()
        print(f"    SOM_d{i} vs mean_diff: {cos:.3f}")

    # Step 3: Dev 数据
    dev_harmful, _ = load_default_datasets(n_harmful=args.n_dev, split="dev", seed=args.seed)
    n = len(dev_harmful)
    print(f"\n[exp_04] {n} dev prompts")

    # Step 4: Baseline
    print(f"\n{'='*60}")
    print(f"BASELINE (n={n})")
    print(f"{'='*60}")
    base_resp = [generate_normal(model, tokenizer, p, max_new_tokens=120)
                 for p in tqdm(dev_harmful, desc="baseline", leave=False)]
    base_stats = batch_classify(base_resp, dev_harmful)
    print(f"  success={base_stats['success_rate']}  refused={base_stats['refused_rate']}")

    # Step 5: Arditi 单方向对比（baseline for comparison）
    from probes.ablate import generate_with_ablation
    print(f"\n{'='*60}")
    print(f"ARDITI SINGLE DIRECTION (mean-diff -> L{ABL_LAYERS}, n={n})")
    print(f"{'='*60}")
    arditi_resp = [generate_with_ablation(model, tokenizer, p, md[17], layers=ABL_LAYERS, max_new_tokens=120)
                   for p in tqdm(dev_harmful, desc="arditi", leave=False)]
    arditi_stats = batch_classify(arditi_resp, dev_harmful)
    print(f"  success={arditi_stats['success_rate']}  refused={arditi_stats['refused_rate']}")

    # Step 6: 贪心搜索最佳 SOM 方向子集
    print(f"\n{'='*60}")
    print(f"GREEDY SEARCH: Best subset of {len(som_dirs)} SOM directions")
    print(f"{'='*60}")
    best_indices, search_history = greedy_direction_search(
        model, tokenizer, som_dirs, dev_harmful, ABL_LAYERS, max_k=args.max_k
    )
    best_som_dirs = [som_dirs[i] for i in best_indices]

    # Step 7: 最终对比 — 全部 SOM 方向 vs 最佳子集 vs Arditi
    print(f"\n{'='*60}")
    print(f"ALL SOM DIRECTIONS ({len(som_dirs)} dirs -> L{ABL_LAYERS})")
    print(f"{'='*60}")
    all_resp = [generate_with_multidir_ablation(model, tokenizer, p, som_dirs, ABL_LAYERS, max_new_tokens=120)
                for p in tqdm(dev_harmful, desc="all_som", leave=False)]
    all_stats = batch_classify(all_resp, dev_harmful)
    print(f"  success={all_stats['success_rate']}  refused={all_stats['refused_rate']}  incoherent={all_stats['incoherent_rate']}")

    # 保存
    output = {
        "model": args.model, "seed": args.seed,
        "abl_layers": ABL_LAYERS,
        "som_info": som_info,
        "n_directions": len(som_dirs),
        "cosine_pairs": cosine_pairs,
        "baseline": {
            "success": base_stats["success"], "n": n,
            "success_rate": base_stats["success_rate"],
            "refused": base_stats["refused"],
        },
        "arditi_single": {
            "success": arditi_stats["success"], "n": n,
            "success_rate": arditi_stats["success_rate"],
            "refused": arditi_stats["refused"],
            "samples": [r[:300] for r in arditi_resp[:3]],
        },
        "greedy_search": {
            "selected_indices": best_indices,
            "history": search_history,
        },
        "all_som_directions": {
            "n_directions": len(som_dirs),
            "success": all_stats["success"], "n": n,
            "success_rate": all_stats["success_rate"],
            "refused": all_stats["refused"], "refused_rate": all_stats["refused_rate"],
            "incoherent": all_stats["incoherent"],
            "samples": [r[:300] for r in all_resp[:5]],
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 摘要
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Method':>25s} | {'Success':>20s} | {'Refused':>10s}")
    print(f"  {'-'*25}-+-{'-'*20}-+-{'-'*10}")
    print(f"  {'Baseline':>25s} | {base_stats['success_rate']:>20s} | {base_stats['refused']}/{n}")
    print(f"  {'Arditi (1 dir)':>25s} | {arditi_stats['success_rate']:>20s} | {arditi_stats['refused']}/{n}")
    if search_history:
        last = search_history[-1]
        print(f"  {f'SOM best-{len(best_indices)} (greedy)':>25s} | {last['success_rate']:>20s} | {last['refused']}/{n}")
    print(f"  {f'SOM all-{len(som_dirs)}':>25s} | {all_stats['success_rate']:>20s} | {all_stats['refused']}/{n}")

    if search_history and search_history[-1]["success"] > arditi_stats["success"]:
        print(f"\n  SOM multi-direction BEATS Arditi single direction.")
    else:
        print(f"\n  SOM multi-direction did NOT beat Arditi. May need tuning.")

    print(f"\n[exp_04] Saved to {args.output}")


if __name__ == "__main__":
    main()
