from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "results" / "pipeline_runs"
EXPERIMENTS_ROOT = ROOT / "experiments"


@dataclass(frozen=True)
class ExperimentSpec:
    script: str
    output_name: str
    stage: str
    default_args: tuple[str, ...] = ()
    arg_aliases: dict[str, str] | None = None
    kind: str = "experiment"
    use_default_output: bool = False


PIPELINE_PRESETS: dict[str, list[ExperimentSpec]] = {
    "gate_scan": [
        ExperimentSpec(
            script="experiments/exp_01_refusal.py",
            output_name="exp01_refusal.json",
            stage="gate_scan",
            arg_aliases={
                "seed": "",
                "n_train": "--n_train",
                "n_eval": "--n_test_full",
                "max_new_tokens": "",
            },
        ),
        ExperimentSpec(
            script="experiments/exp_01b_cross_layer.py",
            output_name="exp01b_cross_layer.json",
            stage="gate_scan",
            arg_aliases={
                "seed": "--seed",
                "n_train": "--n_train",
                "n_eval": "--n_dev",
                "max_new_tokens": "",
            },
        ),
        ExperimentSpec(
            script="experiments/exp_19_l17_l23_late_impact.py",
            output_name="exp19_l17_l23_late_impact.json",
            stage="gate_scan",
            arg_aliases={
                "seed": "--seed",
                "n_train": "--n_train_exec",
                "n_eval": "--n_per_group",
                "max_new_tokens": "--max_new_tokens",
            },
        ),
    ],
    "family_map": [
        ExperimentSpec(
            script="experiments/exp_27_vector_effect_atlas.py",
            output_name="exp27_vector_effect_atlas.json",
            stage="family_map",
            arg_aliases={
                "seed": "--seed",
                "n_train": "--n_train_exec",
                "n_eval": "--n_eval_per_group",
                "max_new_tokens": "--max_new_tokens",
            },
        ),
        ExperimentSpec(
            script="experiments/exp_28_detect_family_causal.py",
            output_name="exp28_detect_family_causal.json",
            stage="family_map",
            arg_aliases={
                "seed": "--seed",
                "n_train": "--n_train_exec",
                "n_eval": "--n_eval_per_group",
                "max_new_tokens": "--max_new_tokens",
            },
        ),
        ExperimentSpec(
            script="experiments/exp_29_pure_detect_disentangle.py",
            output_name="exp29_pure_detect_disentangle.json",
            stage="family_map",
            arg_aliases={
                "seed": "--seed",
                "n_train": "--n_train_exec",
                "n_eval": "--n_eval_per_group",
                "max_new_tokens": "--max_new_tokens",
            },
        ),
        ExperimentSpec(
            script="experiments/exp_30_detect_signed_sweep.py",
            output_name="exp30_detect_signed_sweep.json",
            stage="family_map",
            arg_aliases={
                "seed": "--seed",
                "n_train": "--n_train_exec",
                "n_eval": "--n_eval_per_group",
                "max_new_tokens": "--max_new_tokens",
            },
        ),
        ExperimentSpec(
            script="experiments/exp_31_generation_step_detect_schedule.py",
            output_name="exp31_generation_step_detect_schedule.json",
            stage="family_map",
            arg_aliases={
                "seed": "--seed",
                "n_train": "--n_train_exec",
                "n_eval": "--n_eval_per_group",
                "max_new_tokens": "--max_new_tokens",
            },
        ),
    ],
    "attack_eval": [
        ExperimentSpec(
            script="experiments/exp_38_whitebox_attack_feasibility.py",
            output_name="exp38_whitebox_attack_feasibility.json",
            stage="attack_eval",
            arg_aliases={
                "seed": "--seed",
                "n_train": "--n_train_exec",
                "n_eval": "--n_eval_per_group",
                "max_new_tokens": "--max_new_tokens",
            },
        ),
        ExperimentSpec(
            script="experiments/exp_39_context_knowledge_bypass.py",
            output_name="exp39_context_knowledge_bypass.json",
            stage="attack_eval",
            arg_aliases={
                "seed": "--seed",
                "n_train": "--n_train_exec",
                "n_eval": "--n_eval_per_group",
                "max_new_tokens": "--max_new_tokens",
            },
        ),
    ],
    "analysis_attack_eval": [
        ExperimentSpec(
            script="analysis/format_attack_reports.py",
            output_name="attack_analysis.done",
            stage="analysis",
            kind="analysis",
        ),
    ],
}
PIPELINE_PRESETS["large_model_attack"] = (
    PIPELINE_PRESETS["gate_scan"][:2]
    + [PIPELINE_PRESETS["gate_scan"][2]]
    + PIPELINE_PRESETS["attack_eval"]
    + PIPELINE_PRESETS["analysis_attack_eval"]
)
PIPELINE_PRESETS["full"] = (
    PIPELINE_PRESETS["gate_scan"]
    + PIPELINE_PRESETS["family_map"]
    + PIPELINE_PRESETS["attack_eval"]
    + PIPELINE_PRESETS["analysis_attack_eval"]
)


def script_text(script_relpath: str) -> str:
    return (ROOT / script_relpath).read_text(encoding="utf-8")


def has_flag(script_relpath: str, flag: str) -> bool:
    text = script_text(script_relpath)
    return f'add_argument("{flag}"' in text or f"add_argument('{flag}'" in text


def infer_common_aliases(script_relpath: str) -> dict[str, str]:
    aliases = {"seed": "", "n_train": "", "n_eval": "", "max_new_tokens": ""}
    flag_candidates = {
        "seed": ["--seed"],
        "n_train": ["--n_train_exec", "--n_train"],
        "n_eval": ["--n_eval_per_group", "--n_per_group", "--n_dev", "--n_test_full"],
        "max_new_tokens": ["--max_new_tokens"],
    }
    for key, candidates in flag_candidates.items():
        for flag in candidates:
            if has_flag(script_relpath, flag):
                aliases[key] = flag
                break
    return aliases


def infer_stage(script_relpath: str) -> str:
    path = Path(script_relpath)
    if path.parent.name in {"family_structure", "causal_topology"}:
        return path.parent.name
    return "all_experiments"


def experiment_sort_key(script_relpath: str) -> tuple[int, str]:
    match = re.search(r"exp_(\d+)", Path(script_relpath).name)
    number = int(match.group(1)) if match else 999
    return (number, script_relpath)


def discover_all_experiments() -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for path in sorted(EXPERIMENTS_ROOT.rglob("exp_*.py"), key=lambda p: experiment_sort_key(str(p.relative_to(ROOT)).replace("\\", "/"))):
        relpath = str(path.relative_to(ROOT)).replace("\\", "/")
        output_name = relpath.replace("/", "__").replace(".py", ".native")
        specs.append(
            ExperimentSpec(
                script=relpath,
                output_name=output_name,
                stage=infer_stage(relpath),
                arg_aliases=infer_common_aliases(relpath),
                use_default_output=True,
            )
        )
    return specs


PIPELINE_PRESETS["all_experiments"] = discover_all_experiments()


def parse_repeated_kv(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE format, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in extra arg: {item}")
        parsed[key] = value
    return parsed


def build_model_slug(model_name: str) -> str:
    slug = model_name.replace("/", "__").replace(":", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in slug)


def build_run_name(preset: str, model_name: str, run_name: str | None) -> str:
    if run_name:
        return run_name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{build_model_slug(model_name)}_{preset}"


def build_command(
    spec: ExperimentSpec,
    *,
    python_bin: str,
    model: str,
    hf_token: str | None,
    output_path: Path,
    seed: int | None,
    n_train: int | None,
    n_eval: int | None,
    max_new_tokens: int | None,
    extra_args: dict[str, str],
) -> list[str]:
    cmd = [python_bin, spec.script]
    if spec.kind == "experiment":
        if has_flag(spec.script, "--model"):
            cmd.extend(["--model", model])
        if has_flag(spec.script, "--shield_audit"):
            cmd.append("--shield_audit")
        if has_flag(spec.script, "--shield_device"):
            cmd.extend(["--shield_device", os.getenv("SUPER_GCG_SHIELD_DEVICE", "auto")])
        if has_flag(spec.script, "--shield_truncate"):
            cmd.extend(["--shield_truncate", os.getenv("SUPER_GCG_SHIELD_TRUNCATE", "500")])
        if has_flag(spec.script, "--output") and not spec.use_default_output:
            cmd.extend(["--output", str(output_path)])
        if has_flag(spec.script, "--output_csv") and not spec.use_default_output:
            csv_path = output_path.with_suffix(".csv")
            cmd.extend(["--output_csv", str(csv_path)])
        if has_flag(spec.script, "--output_jsonl") and not spec.use_default_output:
            jsonl_path = output_path.with_suffix(".jsonl")
            cmd.extend(["--output_jsonl", str(jsonl_path)])
        if has_flag(spec.script, "--output_blind") and not spec.use_default_output:
            blind_path = output_path.with_name(output_path.stem + "_blind.json")
            cmd.extend(["--output_blind", str(blind_path)])
        if hf_token and has_flag(spec.script, "--hf_token"):
            cmd.extend(["--hf_token", hf_token])
    else:
        exp38_json = output_path.parent / "exp38_whitebox_attack_feasibility.json"
        exp39_json = output_path.parent / "exp39_context_knowledge_bypass.json"
        exp38_md = output_path.parent / "exp38_summary.md"
        exp39_md = output_path.parent / "exp39_generations.md"
        cmd.extend(
            [
                "--exp38-input",
                str(exp38_json),
                "--exp38-output",
                str(exp38_md),
                "--exp39-input",
                str(exp39_json),
                "--exp39-output",
                str(exp39_md),
            ]
        )
    alias_map = spec.arg_aliases or {}
    common_values = {
        "seed": seed,
        "n_train": n_train,
        "n_eval": n_eval,
        "max_new_tokens": max_new_tokens,
    }
    for key, value in common_values.items():
        alias = alias_map.get(key, "")
        if spec.kind == "experiment" and alias and value is not None:
            cmd.extend([alias, str(value)])
    for key, value in extra_args.items():
        if spec.kind == "experiment":
            cmd.extend([key, value])
    cmd.extend(spec.default_args)
    return cmd


def print_preset_table() -> None:
    for preset_name, specs in PIPELINE_PRESETS.items():
        print(f"{preset_name}:")
        for spec in specs:
            print(f"  - {spec.script}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the current Super GCG experiment stack as a reproducible pipeline."
    )
    parser.add_argument("--preset", default="large_model_attack", choices=sorted(PIPELINE_PRESETS))
    parser.add_argument("--model", default="google/gemma-3-1b-it")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Append raw KEY=VALUE pairs. Use this only for script-specific options.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    args = parser.parse_args()

    if args.list_presets:
        print_preset_table()
        return 0

    preset_specs = PIPELINE_PRESETS[args.preset]
    extra_args = parse_repeated_kv(args.extra_arg)

    run_name = build_run_name(args.preset, args.model, args.run_name)
    results_root = Path(args.results_root).resolve()
    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "preset": args.preset,
        "model": args.model,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "cwd": str(ROOT),
        "python_bin": args.python_bin,
        "hf_token_provided": bool(args.hf_token),
        "extra_args": extra_args,
        "experiments": [],
    }

    for idx, spec in enumerate(preset_specs, start=1):
        output_path = run_dir / spec.output_name
        cmd = build_command(
            spec,
            python_bin=args.python_bin,
            model=args.model,
            hf_token=args.hf_token,
            output_path=output_path,
            seed=args.seed,
            n_train=args.n_train,
            n_eval=args.n_eval,
            max_new_tokens=args.max_new_tokens,
            extra_args=extra_args,
        )
        manifest["experiments"].append(
            {
                "index": idx,
                "stage": spec.stage,
                "script": spec.script,
                "output": str(output_path),
                "use_default_output": spec.use_default_output,
                "command": cmd,
            }
        )

    manifest_path = run_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[pipeline] preset={args.preset}")
    print(f"[pipeline] model={args.model}")
    print(f"[pipeline] run_dir={run_dir}")
    print(f"[pipeline] manifest={manifest_path}")

    for item in manifest["experiments"]:
        command = item["command"]
        pretty = " ".join(shlex.quote(part) for part in command)
        print(f"\n[pipeline] stage={item['stage']} script={item['script']}")
        print(f"[pipeline] cmd: {pretty}")
        if args.dry_run:
            continue
        env = os.environ.copy()
        env.setdefault("SUPER_GCG_ENABLE_SHIELD_AUDIT", "1")
        env.setdefault("SUPER_GCG_SHIELD_DEVICE", "auto")
        env.setdefault("SUPER_GCG_SHIELD_TRUNCATE", "500")
        env.setdefault("SUPER_GCG_SHIELD_SUCCESS_KEY", "direct_danger_fewshot")
        env.setdefault("SUPER_GCG_SHIELD_SUCCESS_THRESHOLD", "0.5")
        completed = subprocess.run(command, cwd=ROOT, check=False, env=env)
        if completed.returncode != 0:
            print(f"[pipeline] failed: {item['script']} exit={completed.returncode}")
            return completed.returncode

    if args.dry_run:
        print("\n[pipeline] dry-run complete")
    else:
        print("\n[pipeline] completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
