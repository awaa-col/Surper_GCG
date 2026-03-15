from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    script: str
    output_name: str
    stage: str
    default_args: tuple[str, ...] = ()
    arg_aliases: dict[str, str] | None = None
    input_bindings: dict[str, str] | None = None
    kind: str = "experiment"
    use_default_output: bool = False


@dataclass(frozen=True)
class StageSpec:
    key: str
    order: int
    title: str
    objective: str
    why_now: str
    automation_mode: str
    shieldgemma_policy: str
    human_review_required: bool
    human_review_points: tuple[str, ...]
    reference_1b_experiments: tuple[str, ...]
    runnable_now: bool
    blocked_reason: str = ""
    experiment_specs: tuple[ExperimentSpec, ...] = ()


PREP_SCOPE_SPECS = [
    ExperimentSpec(
        script="experiments/exp_16_safe_response_dictionary.py",
        output_name="exp16_safe_response_dictionary_full.json",
        stage="prep_scope",
        arg_aliases={
            "seed": "--seed",
            "n_train": "--n_train_exec",
            "n_eval": "--n_per_group",
            "max_new_tokens": "--max_new_tokens",
            "scope_top_k_family": "",
            "min_group_size": "--min_group_size",
        },
    ),
    ExperimentSpec(
        script="experiments/exp_17_gemma_scope_feature_probe.py",
        output_name="exp17_gemma_scope_feature_probe_full.json",
        stage="prep_scope",
        arg_aliases={
            "seed": "--seed",
            "n_train": "",
            "n_eval": "",
            "max_new_tokens": "",
            "scope_top_k_family": "",
            "min_group_size": "--min_group_size",
        },
        input_bindings={
            "--input": "exp16_full",
        },
    ),
]


EVAL_CALIBRATION_SPECS = [
    ExperimentSpec(
        script="experiments/exp_11_review_pack.py",
        output_name="exp11_review_pack.jsonl",
        stage="eval_calibration",
        arg_aliases={
            "seed": "",
            "n_train": "",
            "n_eval": "",
            "max_new_tokens": "",
            "scope_top_k_family": "",
            "min_group_size": "",
        },
    ),
]


BASELINE_DIAGNOSIS_SPECS = [
    ExperimentSpec(
        script="experiments/exp_00_diagnosis.py",
        output_name="exp00_diagnosis.json",
        stage="baseline_diagnosis",
        arg_aliases={
            "seed": "",
            "n_train": "",
            "n_eval": "",
            "max_new_tokens": "",
            "scope_top_k_family": "",
            "min_group_size": "",
        },
    ),
]


GATE_DISCOVERY_SPECS = [
    ExperimentSpec(
        script="experiments/exp_40_gate_discovery.py",
        output_name="exp40_gate_discovery.json",
        stage="gate_discovery",
        arg_aliases={
            "seed": "--seed",
            "n_train": "--n_train",
            "n_eval": "--n_eval",
            "max_new_tokens": "--max_new_tokens",
            "scope_top_k_family": "",
            "min_group_size": "",
        },
    ),
]


FAMILY_MAP_CORE_SPECS = [
    ExperimentSpec(
        script="experiments/exp_27_vector_effect_atlas.py",
        output_name="exp27_vector_effect_atlas.json",
        stage="family_map",
        arg_aliases={
            "seed": "--seed",
            "n_train": "--n_train_exec",
            "n_eval": "--n_eval_per_group",
            "max_new_tokens": "--max_new_tokens",
            "scope_top_k_family": "--scope_top_k_family",
            "min_group_size": "",
        },
        input_bindings={
            "--exp17_input": "exp17_full",
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
            "scope_top_k_family": "--scope_top_k_family",
            "min_group_size": "",
        },
        input_bindings={
            "--exp17_input": "exp17_full",
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
            "scope_top_k_family": "--scope_top_k_family",
            "min_group_size": "",
        },
        input_bindings={
            "--exp17_input": "exp17_full",
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
            "scope_top_k_family": "--scope_top_k_family",
            "min_group_size": "",
        },
        input_bindings={
            "--exp17_input": "exp17_full",
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
            "scope_top_k_family": "--scope_top_k_family",
            "min_group_size": "",
        },
        input_bindings={
            "--exp17_input": "exp17_full",
        },
    ),
]


PIPELINE_STAGES: dict[str, StageSpec] = {
    "eval_calibration": StageSpec(
        key="eval_calibration",
        order=0,
        title="Eval Calibration",
        objective=(
            "Unify 12B review artifacts and establish a ShieldGemma-first audit base "
            "before any mechanism claims."
        ),
        why_now=(
            "Every later stage depends on stable labels and on distinguishing true "
            "unsafe release from degeneration."
        ),
        automation_mode="hybrid",
        shieldgemma_policy=(
            "ShieldGemma is the primary safety label source; heuristic labels may only "
            "assist review-pack prioritization."
        ),
        human_review_required=True,
        human_review_points=(
            "Audit a stratified sample to verify ShieldGemma aligns with semantic risk.",
            "Confirm the project-wide manual label rubric for hard refusal, soft refusal, resource redirect, safe educational, unsafe with disclaimer, direct unsafe, and incoherent.",
        ),
        reference_1b_experiments=("Exp11",),
        runnable_now=True,
        experiment_specs=tuple(EVAL_CALIBRATION_SPECS),
    ),
    "baseline_diagnosis": StageSpec(
        key="baseline_diagnosis",
        order=1,
        title="Baseline Diagnosis",
        objective=(
            "Map the raw 12B refusal profile, prompt-format sensitivity, and obvious "
            "jailbreak seams without importing 1B layer priors."
        ),
        why_now=(
            "This is the first bottom-theory step: confirm that 12B has a stable "
            "refusal object worth localizing."
        ),
        automation_mode="hybrid",
        shieldgemma_policy=(
            "All generated outputs are audited with ShieldGemma; no heuristic headline "
            "rate may replace that audit."
        ),
        human_review_required=True,
        human_review_points=(
            "Inspect high-risk and low-confidence samples to separate true unsafe behavior from malformed outputs.",
            "Write a short baseline memo before moving to gate discovery.",
        ),
        reference_1b_experiments=("Exp00",),
        runnable_now=True,
        experiment_specs=tuple(BASELINE_DIAGNOSIS_SPECS),
    ),
    "gate_discovery": StageSpec(
        key="gate_discovery",
        order=2,
        title="Gate Discovery",
        objective=(
            "Discover whether 12B has a refusal-like causal gate, and where it lives, "
            "without assuming L17 or any fixed layer pair."
        ),
        why_now=(
            "All later detect/late-family theory depends on whether a dominant gate-like "
            "structure exists at all."
        ),
        automation_mode="hybrid",
        shieldgemma_policy=(
            "Candidate layers must be ranked by ShieldGemma-audited behavior shifts plus "
            "degeneration checks, not by heuristics alone."
        ),
        human_review_required=True,
        human_review_points=(
            "Manually inspect top candidate layers and destructive counterexamples.",
            "Reject candidates that only increase nonsense or repetition.",
        ),
        reference_1b_experiments=("Exp01", "Exp01-scan"),
        runnable_now=True,
        blocked_reason="",
        experiment_specs=tuple(GATE_DISCOVERY_SPECS),
    ),
    "cross_layer_refinement": StageSpec(
        key="cross_layer_refinement",
        order=3,
        title="Cross-Layer Refinement",
        objective=(
            "Test whether 12B gate structure is single-layer, pairwise cooperative, or "
            "more distributed."
        ),
        why_now=(
            "The transfer structure determines which causal story is even legal to write."
        ),
        automation_mode="hybrid",
        shieldgemma_policy=(
            "Use ShieldGemma and manual sample review to separate real release from "
            "degeneration in transfer and pair tests."
        ),
        human_review_required=True,
        human_review_points=(
            "Blind-review top single-layer and pair candidates before fixing a mainline intervention.",
        ),
        reference_1b_experiments=("Exp01b", "Exp01-blind"),
        runnable_now=False,
        blocked_reason=(
            "Current transfer logic is still organized around the historical L17/L23 story."
        ),
    ),
    "detect_discovery": StageSpec(
        key="detect_discovery",
        order=4,
        title="Detect Discovery",
        objective=(
            "Re-test whether detect-like structure exists in 12B and whether it is "
            "distinct from the execution gate."
        ),
        why_now=(
            "This is part of the bottom theory; detect cannot be inherited from the 1B "
            "proxy story."
        ),
        automation_mode="hybrid",
        shieldgemma_policy=(
            "Behavioral interpretation remains ShieldGemma-first; geometric candidates "
            "without reliable behavior meaning stay provisional."
        ),
        human_review_required=True,
        human_review_points=(
            "Review the strongest detect-like candidates and compare them with gate candidates.",
        ),
        reference_1b_experiments=("Exp05", "Exp15"),
        runnable_now=False,
        blocked_reason=(
            "The existing detect workflow derives from the 1B gate story and must be reframed around 12B candidates."
        ),
    ),
    "late_safe_response_discovery": StageSpec(
        key="late_safe_response_discovery",
        order=5,
        title="Late Safe-Response Discovery",
        objective=(
            "Find out whether 12B still has a late safe-response organization region, "
            "and what families can be named without overclaiming."
        ),
        why_now=(
            "Only after baseline and gate theory exist can late-layer structure be interpreted safely."
        ),
        automation_mode="hybrid",
        shieldgemma_policy=(
            "Family discovery must start from ShieldGemma-audited natural safe responses, "
            "not heuristic soft-refusal buckets."
        ),
        human_review_required=True,
        human_review_points=(
            "Validate family naming and inspect cross-topic semantic consistency.",
        ),
        reference_1b_experiments=("Exp13", "Exp16", "Exp17"),
        runnable_now=False,
        blocked_reason=(
            "The current Exp16/17 path assumes historical exec-ablation conditions and cannot define 12B late families yet."
        ),
    ),
    "candidate_quantification": StageSpec(
        key="candidate_quantification",
        order=6,
        title="Candidate Quantification",
        objective="Quantify 12B candidate gates and families only after discovery is complete.",
        why_now=(
            "Quantification is meaningless if discovery still assumes 1B layers or vectors."
        ),
        automation_mode="mostly_auto",
        shieldgemma_policy=(
            "Quantitative ranking must stay tied to ShieldGemma-audited behavior outcomes."
        ),
        human_review_required=False,
        human_review_points=(),
        reference_1b_experiments=("Exp18",),
        runnable_now=False,
        blocked_reason="The current quantification script hardcodes L17 as the main gate candidate.",
    ),
    "minimal_causal_closure": StageSpec(
        key="minimal_causal_closure",
        order=7,
        title="Minimal Causal Closure",
        objective=(
            "Connect the 12B upstream gate story with downstream late-family behavior in "
            "the smallest defensible causal chain."
        ),
        why_now="This is the first stage where a mechanism narrative becomes publishable.",
        automation_mode="hybrid",
        shieldgemma_policy=(
            "Every closure claim must be backed by ShieldGemma-audited outputs and manual inspection of key conditions."
        ),
        human_review_required=True,
        human_review_points=(
            "Read key baseline/gate-only/coop-only/gate-plus-coop samples before accepting a closure story.",
        ),
        reference_1b_experiments=("Exp19",),
        runnable_now=False,
        blocked_reason="The historical closure script fixes target layers and late layers from 1B.",
    ),
    "robustness": StageSpec(
        key="robustness",
        order=8,
        title="Robustness",
        objective=(
            "Re-run only already-discovered 12B mechanisms across seeds, splits, and topics."
        ),
        why_now="This stage should never run before a 12B-specific closure exists.",
        automation_mode="mostly_auto",
        shieldgemma_policy="Robustness summaries use ShieldGemma metrics as the default headline.",
        human_review_required=True,
        human_review_points=(
            "Inspect only disagreement and failure cases; do not manually re-label everything.",
        ),
        reference_1b_experiments=("Exp12",),
        runnable_now=False,
        blocked_reason="There is no 12B-specific closure to stress-test yet.",
    ),
    "attack_acceptance": StageSpec(
        key="attack_acceptance",
        order=9,
        title="Attack Acceptance",
        objective="Use attacks only as acceptance tests after 12B mechanism closure is known.",
        why_now="Attack experiments are not allowed to carry the burden of discovery.",
        automation_mode="hybrid",
        shieldgemma_policy=(
            "ShieldGemma remains the acceptance metric, with manual review on apparent successes."
        ),
        human_review_required=True,
        human_review_points=(
            "Read all apparent attack-success samples and verify they exploit the intended mechanism rather than a seam or degeneration.",
        ),
        reference_1b_experiments=("Exp38", "Exp39"),
        runnable_now=False,
        blocked_reason="No 12B mechanism closure exists yet, so attack acceptance would be premature.",
    ),
}


PIPELINE_PRESETS: dict[str, tuple[str, ...]] = {
    "eval_calibration": ("eval_calibration",),
    "baseline_diagnosis": ("baseline_diagnosis",),
    "gate_discovery_bootstrap": ("baseline_diagnosis", "gate_discovery"),
    "mechanism_discovery_foundation": ("eval_calibration",),
    "theory_rebuild_bootstrap": ("eval_calibration", "baseline_diagnosis"),
}


def flatten_stage_specs(stage_keys: tuple[str, ...]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for stage_key in stage_keys:
        specs.extend(PIPELINE_STAGES[stage_key].experiment_specs)
    return specs


def print_preset_table() -> None:
    for preset_name, stage_keys in PIPELINE_PRESETS.items():
        print(f"{preset_name}:")
        for stage_key in stage_keys:
            stage = PIPELINE_STAGES[stage_key]
            print(f"  - [{stage.key}] {stage.title}")
            for spec in stage.experiment_specs:
                print(f"      * {spec.script}")


def print_stage_table() -> None:
    for stage in sorted(PIPELINE_STAGES.values(), key=lambda item: item.order):
        status = "runnable" if stage.runnable_now else "blocked"
        print(f"{stage.order}. {stage.key} [{status}]")
        print(f"   title: {stage.title}")
        print(f"   objective: {stage.objective}")
        print(f"   automation: {stage.automation_mode}")
        print(f"   human_review_required: {stage.human_review_required}")
        if stage.reference_1b_experiments:
            print(f"   1b_reference: {', '.join(stage.reference_1b_experiments)}")
        if stage.blocked_reason:
            print(f"   blocked_reason: {stage.blocked_reason}")


def render_stage_summary(stage_keys: tuple[str, ...]) -> str:
    lines = [
        "# Pipeline Stage Summary",
        "",
        "This run follows the 12B-first mechanism-rebuild order. "
        "1B experiments are references for workflow only, not layer priors.",
        "",
    ]
    for stage in sorted((PIPELINE_STAGES[key] for key in stage_keys), key=lambda item: item.order):
        lines.append(f"## Stage {stage.order}: {stage.title}")
        lines.append("")
        lines.append(f"- Key: `{stage.key}`")
        lines.append(f"- Objective: {stage.objective}")
        lines.append(f"- Why now: {stage.why_now}")
        lines.append(f"- Automation mode: {stage.automation_mode}")
        lines.append(f"- ShieldGemma policy: {stage.shieldgemma_policy}")
        lines.append(
            "- 1B reference only: "
            + (", ".join(stage.reference_1b_experiments) if stage.reference_1b_experiments else "none")
        )
        lines.append(f"- Runnable now: {'yes' if stage.runnable_now else 'no'}")
        if stage.blocked_reason:
            lines.append(f"- Blocked reason: {stage.blocked_reason}")
        if stage.human_review_required:
            lines.append("- Human review checkpoints:")
            for item in stage.human_review_points:
                lines.append(f"  - {item}")
        else:
            lines.append("- Human review checkpoints: none required by default")
        if stage.experiment_specs:
            lines.append("- Pipeline experiments:")
            for spec in stage.experiment_specs:
                lines.append(f"  - `{spec.script}` -> `{spec.output_name}`")
        else:
            lines.append("- Pipeline experiments: none wired yet")
        lines.append("")
    return "\n".join(lines)
