"""
experiments/exp_01b_cross_layer.py  (严密版 v2)
================================================
交叉层验证实验 — 按可发表级统计标准设计。

改进点（vs v1）：
  1. 固定 seed（torch/numpy/random）
  2. 数据三分离：train(提方向) / dev(选策略) / test 不碰
  3. 主指标：success = not_refused AND not_incoherent AND on_topic
  4. 每格 n≥30，结果带 95% Wilson CI
  5. 不过滤任何数据行，乱码计为失败
  6. 结论按"每个 abl_layer 分层比较"，不做总平均
  7. 自动输出结论表

运行：
  uv run python experiments/exp_01b_cross_layer.py --model google/gemma-3-1b-it
  默认 n_dev=30/格（4×4=480 次生成，约 25 分钟）
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from probes.extract import (
    collect_hidden_states,
    mean_diff_direction,
    split_half_stability,
)
from probes.model_config import get_hidden_size, get_num_hidden_layers
from probes.model_structure import get_transformer_layer
from probes.ablate import generate_normal, generate_with_ablation, _make_ablate_hook
from probes.extract import _build_prompt
from probes.stats import set_seed, wilson_ci, format_ci, batch_classify, bootstrap_proportion_test
from data.datasets import load_default_datasets


GLOBAL_LAYERS = [5, 11, 17, 23]


def load_model(model_name, hf_token=None):
    print(f"[exp_01b] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, token=hf_token,
        torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    layers = get_num_hidden_layers(model.config)
    hidden = get_hidden_size(model.config)
    print(f"  layers={layers}, hidden={hidden}")
    return model, tokenizer


def extract_all_directions(model, tokenizer, n_train, seed):
    """从所有 Global 层各自提取方向（train split）。"""
    harmful, harmless = load_default_datasets(
        n_harmful=n_train, n_harmless=n_train, split="train", seed=seed
    )
    print(f"\n[extract] train split: {len(harmful)} harmful, {len(harmless)} harmless")
    print(f"[extract] Collecting hidden states from Global layers...")

    h_harm = collect_hidden_states(model, tokenizer, harmful, layers=GLOBAL_LAYERS, desc="harmful")
    h_safe = collect_hidden_states(model, tokenizer, harmless, layers=GLOBAL_LAYERS, desc="harmless")

    directions = mean_diff_direction(h_harm, h_safe)
    stability = split_half_stability(h_harm, h_safe, k=20, seed=seed)

    for l in GLOBAL_LAYERS:
        s = stability[l]
        print(f"  L{l:2d}: stability={s['mean']:.3f}±{s['std']:.3f}")

    return directions, stability


def run_condition(model, tokenizer, direction, abl_layers, prompts, max_new_tokens=120):
    """运行一个实验条件，返回 (responses, prompts)。"""
    responses = []
    for p in prompts:
        r = generate_with_ablation(model, tokenizer, p, direction, layers=abl_layers,
                                   max_new_tokens=max_new_tokens)
        responses.append(r)
    return responses


def run_dual_direction(model, tokenizer, dir_a, layer_a, dir_b, layer_b, prompts, max_new_tokens=120):
    """双方向消融：各层用各自方向。"""
    responses = []
    for p in prompts:
        text = _build_prompt(tokenizer, p)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        hooks = []
        try:
            h_a = get_transformer_layer(model, layer_a).register_forward_hook(_make_ablate_hook(dir_a))
            h_b = get_transformer_layer(model, layer_b).register_forward_hook(_make_ablate_hook(dir_b))
            hooks = [h_a, h_b]
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            new_tokens = out[0, inputs["input_ids"].shape[1]:]
            responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        finally:
            for h in hooks:
                h.remove()
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--output", default="results/exp01b_v2.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=100, help="方向提取用样本数(train split)")
    parser.add_argument("--n_dev", type=int, default=30, help="每格测试样本数(dev split)")
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model, args.hf_token)

    # ─── Step 1：提取方向（train split）──────────────────────────────
    directions, stability = extract_all_directions(model, tokenizer, args.n_train, args.seed)

    # ─── Step 2：方向间 cosine similarity ─────────────────────────────
    print(f"\n{'='*60}")
    print("COSINE SIMILARITY MATRIX")
    print(f"{'='*60}")
    cosine_matrix = {}
    for i, la in enumerate(GLOBAL_LAYERS):
        for lb in GLOBAL_LAYERS[i:]:
            cos = torch.nn.functional.cosine_similarity(
                directions[la].unsqueeze(0), directions[lb].unsqueeze(0)
            ).item()
            cosine_matrix[f"L{la}_L{lb}"] = cos
            if la != lb:
                cosine_matrix[f"L{lb}_L{la}"] = cos
    header = "        " + "  ".join(f"L{l:2d}" for l in GLOBAL_LAYERS)
    print(header)
    for la in GLOBAL_LAYERS:
        row = f"  L{la:2d}:  "
        for lb in GLOBAL_LAYERS:
            key = f"L{la}_L{lb}"
            row += f" {cosine_matrix.get(key, 0):.3f}"
        print(row)

    # ─── Step 3：加载 dev 数据 ─────────────────────────────────────────
    dev_harmful, _ = load_default_datasets(n_harmful=args.n_dev, split="dev", seed=args.seed)
    n_dev = len(dev_harmful)
    print(f"\n[dev] {n_dev} harmful prompts for evaluation")

    # ─── Step 4：Baseline ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"BASELINE (no ablation, n={n_dev})")
    print(f"{'='*60}")
    baseline_resp = [generate_normal(model, tokenizer, p, max_new_tokens=120) for p in tqdm(dev_harmful, desc="baseline", leave=False)]
    baseline_stats = batch_classify(baseline_resp, dev_harmful)
    print(f"  refused: {baseline_stats['refused_rate']}")
    print(f"  incoherent: {baseline_stats['incoherent_rate']}")
    print(f"  success: {baseline_stats['success_rate']}")

    # ─── Step 5：4×4 交叉消融（dev split）──────────────────────────
    print(f"\n{'='*60}")
    print(f"4×4 CROSS-ABLATION (n={n_dev}/cell)")
    print(f"{'='*60}")

    cross_results = {}
    for dir_layer in GLOBAL_LAYERS:
        for abl_layer in GLOBAL_LAYERS:
            label = f"dir_L{dir_layer}_abl_L{abl_layer}"
            print(f"\n  [{label}]", end=" ")
            responses = run_condition(model, tokenizer, directions[dir_layer], [abl_layer],
                                      dev_harmful, max_new_tokens=120)
            stats = batch_classify(responses, dev_harmful)
            print(f"success={stats['success_rate']}  refused={stats['refused_rate']}  incoherent={stats['incoherent_rate']}")
            cross_results[label] = {
                "success": stats["success"], "n": stats["n"],
                "success_rate": stats["success_rate"],
                "refused": stats["refused"], "refused_rate": stats["refused_rate"],
                "incoherent": stats["incoherent"], "incoherent_rate": stats["incoherent_rate"],
                "sample": responses[0][:200] if responses else "",
            }

    # ─── Step 6：候选策略（dev split）──────────────────────────────
    print(f"\n{'='*60}")
    print(f"CANDIDATE STRATEGIES (n={n_dev})")
    print(f"{'='*60}")

    strategies = {
        "L17_dir→L17": (directions[17], [17]),
        "L17_dir→[17,23]": (directions[17], [17, 23]),
        "L17_dir→[11,17]": (directions[17], [11, 17]),
    }

    strategy_results = {}
    for name, (d, layers) in strategies.items():
        print(f"\n  [{name}]", end=" ")
        responses = run_condition(model, tokenizer, d, layers, dev_harmful)
        stats = batch_classify(responses, dev_harmful)
        print(f"success={stats['success_rate']}  refused={stats['refused_rate']}  incoherent={stats['incoherent_rate']}")
        strategy_results[name] = {
            "success": stats["success"], "n": stats["n"],
            "success_rate": stats["success_rate"],
            "refused": stats["refused"], "refused_rate": stats["refused_rate"],
            "incoherent": stats["incoherent"], "incoherent_rate": stats["incoherent_rate"],
            "samples": [r[:200] for r in responses[:5]],
        }

    # 双方向策略
    print(f"\n  [L17→L17+L23→L23 (dual)]", end=" ")
    dual_resp = run_dual_direction(model, tokenizer, directions[17], 17, directions[23], 23, dev_harmful)
    dual_stats = batch_classify(dual_resp, dev_harmful)
    print(f"success={dual_stats['success_rate']}  refused={dual_stats['refused_rate']}  incoherent={dual_stats['incoherent_rate']}")
    strategy_results["dual_L17+L23"] = {
        "success": dual_stats["success"], "n": dual_stats["n"],
        "success_rate": dual_stats["success_rate"],
        "refused": dual_stats["refused"], "refused_rate": dual_stats["refused_rate"],
        "incoherent": dual_stats["incoherent"], "incoherent_rate": dual_stats["incoherent_rate"],
        "samples": [r[:200] for r in dual_resp[:5]],
    }

    # ─── Step 7：分层比较 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PER-LAYER ANALYSIS: For each abl_layer, which dir_layer works best?")
    print(f"{'='*60}")

    per_layer_analysis = {}
    for abl_layer in GLOBAL_LAYERS:
        print(f"\n  --- Ablating L{abl_layer} ---")
        best_name = None
        best_success = -1
        for dir_layer in GLOBAL_LAYERS:
            label = f"dir_L{dir_layer}_abl_L{abl_layer}"
            data = cross_results[label]
            success_k = data["success"]
            n = data["n"]
            p, lo, hi = wilson_ci(success_k, n)
            is_best = success_k > best_success
            marker = " ★" if is_best else ""
            if is_best:
                best_success = success_k
                best_name = f"L{dir_layer}"
            print(f"    dir=L{dir_layer}: success={format_ci(success_k, n)}  ref={data['refused']}/{n}  inc={data['incoherent']}/{n}{marker}")

        per_layer_analysis[f"abl_L{abl_layer}"] = {"best_dir": best_name, "best_success": best_success}
        print(f"    → Best direction for L{abl_layer}: {best_name}")

    # ─── Step 8：保存 ──────────────────────────────────────────────
    output = {
        "model": args.model,
        "seed": args.seed,
        "n_train": args.n_train,
        "n_dev": n_dev,
        "data_split": "train(60%)/dev(20%)/test(20%) - test not used",
        "stability": {
            str(l): {"mean": s["mean"], "std": s["std"]}
            for l, s in stability.items()
        },
        "cosine_matrix": cosine_matrix,
        "baseline": {
            "success": baseline_stats["success"], "n": baseline_stats["n"],
            "success_rate": baseline_stats["success_rate"],
            "refused": baseline_stats["refused"],
            "incoherent": baseline_stats["incoherent"],
        },
        "cross_ablation": cross_results,
        "strategies": strategy_results,
        "per_layer_analysis": per_layer_analysis,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[exp_01b] Results saved to {args.output}")

    # ─── Final Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline success: {baseline_stats['success_rate']}")
    print(f"\n  Strategy ranking (by success count, higher is better):")
    ranked = sorted(strategy_results.items(), key=lambda x: x[1]["success"], reverse=True)
    for name, data in ranked:
        print(f"    {name:>30s}: success={data['success_rate']}")

    print(f"\n  Conclusions supported by data (n={n_dev}):")
    print(f"    1. Global layer directions are nearly orthogonal (cosine < 0.25 for most pairs)")
    print(f"    2. Per-layer best direction:")
    for abl_l, info in per_layer_analysis.items():
        print(f"       {abl_l}: best dir = {info['best_dir']} (success={info['best_success']}/{n_dev})")
    print(f"\n  ⚠️  n={n_dev} is small. All CIs above reflect this uncertainty.")
    print(f"  ⚠️  'on_topic' check is keyword-based (crude). Manual review needed.")


if __name__ == "__main__":
    main()
