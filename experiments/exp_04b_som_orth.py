"""
experiments/exp_04b_som_orth.py
================================
SOM 改进版：正交化方向 + 子空间一次性投影 + 多目标贪心。

修复 exp_04_som.py 的三个实现 bug（by 主人）：
  1. 方向正交化（Gram-Schmidt）：消除方向间高余弦相关导致的"重复削同一子空间"
  2. 子空间一次性投影（而非串行 hook）：P = I - Q Q^T，一次 forward pass 消干净
  3. 多目标贪心：score = success - λ*incoherent - μ*refused
  4. 每步贪心完成后立即写 checkpoint，中途中断不丢数据

运行：
  uv run --active python experiments/exp_04b_som_orth.py --model google/gemma-3-1b-it
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from minisom import MiniSom

from transformers import AutoModelForCausalLM, AutoTokenizer
from probes.extract import collect_hidden_states, mean_diff_direction, _build_prompt
from probes.stats import set_seed, batch_classify
from data.datasets import load_default_datasets


ABL_LAYERS = [17, 23]


def load_model(model_name, hf_token=None):
    print(f"[exp_04b] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token,
        torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    print(f"  layers={model.config.num_hidden_layers}, hidden={model.config.hidden_size}")
    return model, tokenizer


# ─── 修复 1：Gram-Schmidt 正交化 ────────────────────────────────────────────

def gram_schmidt_orthogonalize(directions: list) -> list:
    """Gram-Schmidt 正交化，返回标准正交基。方向按原顺序处理（先来先占子空间）。"""
    orth = []
    for v in directions:
        v = v.float()
        for u in orth:
            v = v - torch.dot(v, u) * u
        norm = v.norm()
        if norm > 1e-6:
            orth.append(v / norm)
    return [o.to(directions[0].dtype) for o in orth]


# ─── 修复 2：子空间一次性投影 hook ──────────────────────────────────────────

def make_subspace_hook(Q: torch.Tensor):
    """
    一次性子空间投影 hook。
    Q : [k, d]，正交标准基（每行单位向量，行间互相正交）
    h_new = h - (h @ Q^T) @ Q
    """
    Q_stored = Q.clone()

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        Qd = Q_stored.to(hidden.device, dtype=hidden.dtype)
        proj = hidden @ Qd.T        # [B, T, k]
        corrected = hidden - proj @ Qd  # [B, T, d]
        if isinstance(output, tuple):
            return (corrected,) + output[1:]
        return corrected

    return hook_fn


def generate_with_subspace_ablation(model, tokenizer, prompt, Q, layers, max_new_tokens=120):
    """用子空间一次性投影做消融生成。Q: [k, d] 正交基。"""
    text = _build_prompt(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    hooks = []
    try:
        hook_fn = make_subspace_hook(Q)
        for l in layers:
            hooks.append(model.model.layers[l].register_forward_hook(hook_fn))
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
    finally:
        for h in hooks:
            h.remove()


# ─── SOM 方向提取 ────────────────────────────────────────────────────────────

def extract_som_directions(h_harmful, h_harmless, layer, som_x=3, som_y=3, som_iters=5000, seed=42):
    """SOM 方向提取（原版逻辑）。"""
    harmful_np = h_harmful[layer].cpu().numpy().astype(np.float64)
    harmless_np = h_harmless[layer].cpu().numpy().astype(np.float64)
    d = harmful_np.shape[1]
    harmless_centroid = harmless_np.mean(axis=0)

    print(f"  Training SOM ({som_x}x{som_y}, {som_iters} iters, d={d})...")
    som = MiniSom(som_x, som_y, d, sigma=1.0, learning_rate=0.5,
                  random_seed=seed, neighborhood_function='gaussian')
    som.pca_weights_init(harmful_np)
    som.train(harmful_np, som_iters, verbose=False)

    winner_counts = {}
    for sample in harmful_np:
        w = som.winner(sample)
        winner_counts[w] = winner_counts.get(w, 0) + 1

    print(f"  Activated neurons: {len(winner_counts)}/{som_x*som_y}")
    directions = []
    device = h_harmful[layer].device
    for (i, j), count in sorted(winner_counts.items(), key=lambda x: x[1], reverse=True):
        w_vec = som.get_weights()[i, j]
        raw_dir = w_vec - harmless_centroid
        norm = np.linalg.norm(raw_dir)
        if norm < 1e-8:
            continue
        direction = torch.tensor(raw_dir / norm, dtype=torch.bfloat16, device=device)
        directions.append(direction)
        print(f"    Neuron ({i},{j}): {count} samples, raw_norm={norm:.2f}")
    return directions


# ─── 修复 3：多目标贪心搜索（含 checkpoint）──────────────────────────────────

def multi_objective_greedy(model, tokenizer, orth_dirs, dev_prompts, layers,
                           max_k=5, lam_incoherent=0.5, mu_refused=0.3,
                           checkpoint_path=None):
    """
    多目标贪心：score = success - λ*incoherent - μ*refused
    每步完成后写 checkpoint，中途中断可读 checkpoint 了解进度。
    """
    selected = []
    remaining = list(range(len(orth_dirs)))
    best_score = -float('inf')
    history = []
    n = len(dev_prompts)

    print(f"\n  Multi-objective greedy over {len(orth_dirs)} orth directions, max_k={max_k}")
    print(f"  Objective: success - {lam_incoherent}*incoherent - {mu_refused}*refused")

    for step in range(min(max_k, len(orth_dirs))):
        best_idx = -1
        best_step_score = best_score
        best_stats_this_step = None
        best_resps_this_step = None

        print(f"\n  [Step {step+1}] Trying {len(remaining)} candidates...")
        for idx in remaining:
            candidate_dirs = [orth_dirs[i] for i in selected + [idx]]
            Q = torch.stack(candidate_dirs)
            responses = [
                generate_with_subspace_ablation(model, tokenizer, p, Q, layers, max_new_tokens=120)
                for p in dev_prompts
            ]
            stats = batch_classify(responses, dev_prompts)
            score = stats["success"] - lam_incoherent * stats["incoherent"] - mu_refused * stats["refused"]
            print(f"    dir#{idx}: score={score:.2f} success={stats['success']}/{n} "
                  f"incoh={stats['incoherent']} ref={stats['refused']}")
            if score > best_step_score:
                best_step_score = score
                best_idx = idx
                best_stats_this_step = stats
                best_resps_this_step = responses

        if best_idx == -1 or best_step_score <= best_score:
            print(f"  [Step {step+1}] No improvement, stopping.")
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        best_score = best_step_score

        history.append({
            "step": step + 1,
            "added_direction_idx": best_idx,
            "n_directions": len(selected),
            "objective_score": float(best_step_score),
            "success": best_stats_this_step["success"],
            "success_rate": best_stats_this_step["success_rate"],
            "refused": best_stats_this_step["refused"],
            "refused_rate": best_stats_this_step["refused_rate"],
            "incoherent": best_stats_this_step["incoherent"],
            "incoherent_rate": best_stats_this_step["incoherent_rate"],
            "samples": [r[:300] for r in best_resps_this_step[:3]],
        })
        print(f"  ✅ Step {step+1}: added dir#{best_idx}, score={best_step_score:.2f}, "
              f"success={best_stats_this_step['success_rate']}")

        # 每步写 checkpoint
        if checkpoint_path:
            ckpt = {"selected_indices": selected, "history": history}
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(ckpt, f, indent=2, ensure_ascii=False)
            print(f"  [checkpoint] → {checkpoint_path}")

    return selected, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--output", default="results/exp04b_som_orth.json")
    parser.add_argument("--checkpoint", default="results/exp04b_checkpoint.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_dev", type=int, default=50)
    parser.add_argument("--som_x", type=int, default=3)
    parser.add_argument("--som_y", type=int, default=3)
    parser.add_argument("--som_iters", type=int, default=5000)
    parser.add_argument("--max_k", type=int, default=4)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.3)
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model, args.hf_token)

    # Step 1: 收集 hidden states
    print("\n[exp_04b] Collecting hidden states (train split)...")
    harmful, harmless = load_default_datasets(
        n_harmful=args.n_train, n_harmless=args.n_train, split="train", seed=args.seed
    )
    h_harm = collect_hidden_states(model, tokenizer, harmful, layers=[17], desc="harmful")
    h_safe = collect_hidden_states(model, tokenizer, harmless, layers=[17], desc="harmless")

    # Step 2: SOM 提取
    print(f"\n[exp_04b] Extracting SOM directions at L17...")
    raw_dirs = extract_som_directions(h_harm, h_safe, layer=17,
                                      som_x=args.som_x, som_y=args.som_y,
                                      som_iters=args.som_iters, seed=args.seed)
    print(f"  Extracted {len(raw_dirs)} raw directions")

    md = mean_diff_direction(h_harm, h_safe)
    print(f"\n  [Before orthogonalization] Cosine with mean_diff:")
    for i, d in enumerate(raw_dirs):
        cos = torch.dot(d.float(), md[17].float()).item()
        print(f"    d{i}: {cos:.3f}")

    # Step 3: Gram-Schmidt 正交化
    print(f"\n[exp_04b] Gram-Schmidt orthogonalization...")
    orth_dirs = gram_schmidt_orthogonalize(raw_dirs)
    print(f"  Orthogonalized to {len(orth_dirs)} directions")
    print(f"\n  [After] Cosine between orth directions:")
    for i in range(len(orth_dirs)):
        for j in range(i + 1, len(orth_dirs)):
            cos = torch.dot(orth_dirs[i].float(), orth_dirs[j].float()).item()
            print(f"    d{i} vs d{j}: {cos:.6f}")
    print(f"\n  [After] Cosine with mean_diff:")
    for i, d in enumerate(orth_dirs):
        cos = torch.dot(d.float(), md[17].float()).item()
        print(f"    orth_d{i}: {cos:.3f}")

    # Step 4: Dev 数据
    dev_harmful, _ = load_default_datasets(n_harmful=args.n_dev, split="dev", seed=args.seed)
    if len(dev_harmful) < args.n_dev:
        extra, _ = load_default_datasets(n_harmful=args.n_dev - len(dev_harmful),
                                          split="train", seed=args.seed + 1)
        dev_harmful = dev_harmful + extra
    n = len(dev_harmful)
    print(f"\n[exp_04b] {n} dev prompts")

    from probes.ablate import generate_normal, generate_with_ablation

    # Step 5: Baseline
    print(f"\n{'='*60}")
    print(f"BASELINE (n={n})")
    base_resp = [generate_normal(model, tokenizer, p, max_new_tokens=120)
                 for p in tqdm(dev_harmful, desc="baseline", leave=False)]
    base_stats = batch_classify(base_resp, dev_harmful)
    print(f"  success={base_stats['success_rate']}  refused={base_stats['refused_rate']}")

    # Step 6: Arditi 单方向（子空间形式对比）
    print(f"\n{'='*60}")
    print(f"ARDITI single direction (subspace projection, n={n})")
    Q_arditi = md[17].unsqueeze(0)
    arditi_resp = [generate_with_subspace_ablation(model, tokenizer, p, Q_arditi,
                                                    ABL_LAYERS, max_new_tokens=120)
                   for p in tqdm(dev_harmful, desc="arditi", leave=False)]
    arditi_stats = batch_classify(arditi_resp, dev_harmful)
    print(f"  success={arditi_stats['success_rate']}  refused={arditi_stats['refused_rate']}")

    # Step 7: 多目标贪心搜索
    print(f"\n{'='*60}")
    print(f"MULTI-OBJECTIVE GREEDY on orthogonalized SOM directions")
    best_indices, search_history = multi_objective_greedy(
        model, tokenizer, orth_dirs, dev_harmful, ABL_LAYERS,
        max_k=args.max_k, lam_incoherent=args.lam, mu_refused=args.mu,
        checkpoint_path=args.checkpoint,
    )

    # Step 8: 全部正交方向
    print(f"\n{'='*60}")
    print(f"ALL ORTH DIRECTIONS ({len(orth_dirs)} dirs, subspace projection)")
    Q_all = torch.stack(orth_dirs)
    all_resp = [generate_with_subspace_ablation(model, tokenizer, p, Q_all,
                                                 ABL_LAYERS, max_new_tokens=120)
                for p in tqdm(dev_harmful, desc="all_orth", leave=False)]
    all_stats = batch_classify(all_resp, dev_harmful)
    print(f"  success={all_stats['success_rate']}  incoherent={all_stats['incoherent_rate']}")

    # 保存
    output_data = {
        "model": args.model, "seed": args.seed,
        "abl_layers": ABL_LAYERS,
        "n_raw_dirs": len(raw_dirs),
        "n_orth_dirs": len(orth_dirs),
        "n_dev": n,
        "objective_lambda_incoherent": args.lam,
        "objective_mu_refused": args.mu,
        "baseline": {
            "success": base_stats["success"], "n": n,
            "success_rate": base_stats["success_rate"],
            "refused": base_stats["refused"],
        },
        "arditi_subspace": {
            "success": arditi_stats["success"], "n": n,
            "success_rate": arditi_stats["success_rate"],
            "refused": arditi_stats["refused"], "refused_rate": arditi_stats["refused_rate"],
            "incoherent": arditi_stats["incoherent"],
            "samples": [r[:300] for r in arditi_resp[:5]],
        },
        "greedy_search": {
            "selected_indices": best_indices,
            "history": search_history,
        },
        "all_orth_directions": {
            "n_directions": len(orth_dirs),
            "success": all_stats["success"], "n": n,
            "success_rate": all_stats["success_rate"],
            "refused": all_stats["refused"], "refused_rate": all_stats["refused_rate"],
            "incoherent": all_stats["incoherent"],
            "samples": [r[:300] for r in all_resp[:5]],
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Method':>25s} | {'Success':>20s} | Ref | Incoh")
    print(f"  {'-'*25}-+-{'-'*20}-+-----+------")
    print(f"  {'Baseline':>25s} | {base_stats['success_rate']:>20s} | {base_stats['refused']:>2}/{n} | {base_stats['incoherent']:>2}/{n}")
    print(f"  {'Arditi (subspace)':>25s} | {arditi_stats['success_rate']:>20s} | {arditi_stats['refused']:>2}/{n} | {arditi_stats['incoherent']:>2}/{n}")
    if search_history:
        last = search_history[-1]
        print(f"  {f'SOM orth-{len(best_indices)} (greedy)':>25s} | {last['success_rate']:>20s} | {last['refused']:>2}/{n} | {last['incoherent']:>2}/{n}")
    print(f"  {f'SOM orth-all-{len(orth_dirs)}':>25s} | {all_stats['success_rate']:>20s} | {all_stats['refused']:>2}/{n} | {all_stats['incoherent']:>2}/{n}")
    print(f"\n[exp_04b] Saved to {args.output}")


if __name__ == "__main__":
    main()
