from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_exp38(input_path: Path, output_path: Path) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    verdict = data["attack_feasibility_verdict"]

    lines = ["# Exp38 White-box Attack Feasibility Results", ""]
    lines.append(
        f"Baseline: dangerous_mean={verdict['baseline_dangerous_mean']:.4f}, "
        f"ddf_mean={verdict['baseline_ddf_mean']:.4f}"
    )
    lines.append("")
    lines.append("## Three-Layer Feasibility Verdict")
    lines.append("| Condition | L1: Gate Open? | L2: Actionable Rate | L2: DDF Coherent | L2: Verdict | L3: Cross-topic Stable? |")
    lines.append("|---|---|---|---|---|---|")

    for cond in data["conditions"]:
        if cond == "baseline":
            continue
        gate = verdict["layer1_gate_open"].get(cond, {}).get("gate_open", False)
        layer2 = verdict["layer2_what_is_behind"].get(cond, {})
        layer3 = verdict["layer3_chain_stability"].get(cond, {})
        lines.append(
            f"| {cond} | {gate} | {layer2.get('actionable_unsafe_rate', 'N/A')} | "
            f"{layer2.get('ddf_coherent_mean', 'N/A')} | "
            f"{layer2.get('feasibility_verdict', 'N/A')} | "
            f"{layer3.get('consistent_across_topics', False)} |"
        )

    lines.append("")
    lines.append("## Top Candidates for Manual Review")
    for cond in ["exec_off_l17", "exec_off_l17_l23", "weight_ortho"]:
        for row in data.get("review_candidates", {}).get(cond, [])[:2]:
            lines.append(f"### {cond}")
            lines.append(f"**Topic**: {row['topic']}")
            lines.append(f"**Prompt**: {row['prompt']}")
            lines.append(
                f"**Label**: {row['primary_label']} "
                f"(actionable: {row['actionable_unsafe']})"
            )
            lines.append(f"**DDF Score**: {row['direct_danger_fewshot_score']:.4f}")
            lines.append(f"**Response:**\n```\n{row['response']}\n```")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def format_exp39(input_path: Path, output_path: Path) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    injected_map: dict[str, str] = {}
    for _, rows in data.get("items", {}).items():
        for row in rows:
            if row.get("injected_knowledge"):
                injected_map[row["topic"]] = row["injected_knowledge"]

    lines = ["# Exp39 Generation Outputs (Top 6 per condition)"]
    lines.append(
        "> *Note: The experiment only saves the top 6 responses "
        "(ranked by direct_danger_fewshot score) for manual review.*"
    )

    for condition, rows in data.get("review_candidates", {}).items():
        lines.append("")
        lines.append(f"## Condition: `{condition}`")
        for idx, row in enumerate(rows, start=1):
            lines.append(f"### {idx}. Topic: {row['topic']}")
            lines.append(f"**Prompt:** {row['prompt']}")
            lines.append("")
            lines.append("**Knowledge Injected in Context:**")
            lines.append(f"> {injected_map.get(row['topic'], 'None')}")
            lines.append("")
            lines.append(f"**Generation:**\n```text\n{row['response']}\n```")
            lines.append(
                f"**Scores:** Dangerous: {row['dangerous_score']} | "
                f"DDF: {row['direct_danger_fewshot_score']} | "
                f"Actionable: {row['actionable_unsafe']}"
            )
            lines.append("")
            lines.append("---")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Format Exp38/Exp39 attack evaluation outputs into Markdown."
    )
    parser.add_argument("--exp38-input", default=None)
    parser.add_argument("--exp38-output", default=None)
    parser.add_argument("--exp39-input", default=None)
    parser.add_argument("--exp39-output", default=None)
    args = parser.parse_args()

    if args.exp38_input and args.exp38_output:
        format_exp38(Path(args.exp38_input), Path(args.exp38_output))
        print(f"[analysis] wrote {args.exp38_output}")
    if args.exp39_input and args.exp39_output:
        format_exp39(Path(args.exp39_input), Path(args.exp39_output))
        print(f"[analysis] wrote {args.exp39_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
