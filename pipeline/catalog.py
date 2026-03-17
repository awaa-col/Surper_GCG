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
    tech_id: str
    order: int
    title: str
    objective: str
    why_now: str
    automation_mode: str
    shieldgemma_policy: str
    human_review_required: bool
    human_review_points: tuple[str, ...]
    experiment_ids: tuple[str, ...]
    depends_on: tuple[str, ...]
    runnable_now: bool
    blocked_reason: str = ""
    experiment_specs: tuple[ExperimentSpec, ...] = ()


T0_EVAL_CALIBRATION_SPECS = [
    ExperimentSpec(
        script="experiments/exp_11_review_pack.py",
        output_name="exp11_review_pack.jsonl",
        stage="t0_eval_calibration",
        arg_aliases={"seed": "", "n_train": "", "n_eval": "", "max_new_tokens": ""},
    ),
]


T1_BASELINE_DIAGNOSIS_SPECS = [
    ExperimentSpec(
        script="experiments/exp_00_diagnosis.py",
        output_name="exp00_diagnosis.json",
        stage="t1_baseline_diagnosis",
        arg_aliases={"seed": "", "n_train": "", "n_eval": "", "max_new_tokens": ""},
    ),
]


T2_GATE_DISCOVERY_SPECS = [
    ExperimentSpec(
        script="experiments/exp_40_gate_discovery.py",
        output_name="exp40_gate_discovery.json",
        stage="t2_gate_discovery",
        arg_aliases={
            "seed": "--seed",
            "n_train": "--n_train",
            "n_eval": "--n_eval",
            "max_new_tokens": "--max_new_tokens",
        },
    ),
]


PIPELINE_STAGES: dict[str, StageSpec] = {
    "t0_eval_calibration": StageSpec(
        key="t0_eval_calibration",
        tech_id="T0",
        order=0,
        title="Eval Calibration",
        objective="Unify the 12B review rubric and establish ShieldGemma-first auditing before any mechanism interpretation.",
        why_now="Every downstream stage depends on stable labels and a reliable split between true unsafe release and degeneration.",
        automation_mode="hybrid",
        shieldgemma_policy="ShieldGemma is the primary safety label source; heuristic labels are non-authoritative helpers only.",
        human_review_required=True,
        human_review_points=(
            "Audit a stratified sample and verify the project label rubric is consistent.",
            "Freeze the manual review policy before baseline diagnosis starts.",
        ),
        experiment_ids=("Exp11",),
        depends_on=(),
        runnable_now=True,
        experiment_specs=tuple(T0_EVAL_CALIBRATION_SPECS),
    ),
    "t1_baseline_diagnosis": StageSpec(
        key="t1_baseline_diagnosis",
        tech_id="T1",
        order=1,
        title="Baseline Diagnosis",
        objective="Map the raw 12B refusal profile and prompt sensitivity without importing historical layer priors.",
        why_now="This is the first real discovery step: confirm the baseline object before searching for a causal gate.",
        automation_mode="hybrid",
        shieldgemma_policy="All baseline outputs stay ShieldGemma-audited; no heuristic headline rate may replace that audit.",
        human_review_required=True,
        human_review_points=(
            "Inspect risky and malformed outputs separately.",
            "Write a baseline memo before advancing to gate discovery.",
        ),
        experiment_ids=("Exp00",),
        depends_on=("t0_eval_calibration",),
        runnable_now=True,
        experiment_specs=tuple(T1_BASELINE_DIAGNOSIS_SPECS),
    ),
    "t2_gate_discovery": StageSpec(
        key="t2_gate_discovery",
        tech_id="T2",
        order=2,
        title="Gate Discovery",
        objective="Search the full layer stack for candidate 12B gate layers without assuming L17 or any fixed pair.",
        why_now="Later detect and late-family claims are illegal until a candidate gate story exists.",
        automation_mode="hybrid",
        shieldgemma_policy="Candidate layers must be ranked by ShieldGemma-audited behavior shifts plus degeneration checks.",
        human_review_required=True,
        human_review_points=(
            "Review top candidate and destructive layers by hand.",
            "Reject layers that only increase incoherence or repetition.",
            "Do not promote any layer to a main gate from Exp40 alone; treat T2 output as a provisional shortlist only.",
        ),
        experiment_ids=("Exp01", "Exp01-scan", "Exp40"),
        depends_on=("t1_baseline_diagnosis",),
        runnable_now=True,
        experiment_specs=tuple(T2_GATE_DISCOVERY_SPECS),
    ),
    "t3_cross_layer_refinement": StageSpec(
        key="t3_cross_layer_refinement",
        tech_id="T3",
        order=3,
        title="Cross-Layer Refinement",
        objective="Test whether the gate is single-layer, pairwise cooperative, or more distributed.",
        why_now="This determines which intervention story is even valid to write.",
        automation_mode="hybrid",
        shieldgemma_policy="Use ShieldGemma plus blind sample review to separate real release from degeneration.",
        human_review_required=True,
        human_review_points=("Blind-review the top single-layer and pair candidates before fixing a mainline intervention.",),
        experiment_ids=("Exp01b", "Exp01-blind"),
        depends_on=("t2_gate_discovery",),
        runnable_now=False,
        blocked_reason="Current transfer logic still assumes the historical L17/L23 story.",
    ),
    "t4_detect_discovery": StageSpec(
        key="t4_detect_discovery",
        tech_id="T4",
        order=4,
        title="Detect Discovery",
        objective="Rediscover detect-like structure in 12B without inheriting the 1B detect proxy.",
        why_now="Detect cannot be a borrowed story; it must be rediscovered around 12B candidates.",
        automation_mode="hybrid",
        shieldgemma_policy="Behavior meaning stays ShieldGemma-first; geometry without stable behavior stays provisional.",
        human_review_required=True,
        human_review_points=("Review the strongest detect-like candidates and compare them against gate candidates.",),
        experiment_ids=("Exp05", "Exp15"),
        depends_on=("t3_cross_layer_refinement",),
        runnable_now=False,
        blocked_reason="The current detect workflow is still derived from the 1B gate story.",
    ),
    "t5_late_safe_response_discovery": StageSpec(
        key="t5_late_safe_response_discovery",
        tech_id="T5",
        order=5,
        title="Late Safe-Response Discovery",
        objective="Rediscover whether 12B has a late safe-response region and what families can be named safely.",
        why_now="Late-layer interpretation is only legal after baseline and gate candidates exist.",
        automation_mode="hybrid",
        shieldgemma_policy="Start from ShieldGemma-audited natural safe responses, not heuristic soft-refusal buckets.",
        human_review_required=True,
        human_review_points=("Validate family naming and inspect cross-topic consistency.",),
        experiment_ids=("Exp13", "Exp16"),
        depends_on=("t3_cross_layer_refinement",),
        runnable_now=False,
        blocked_reason="Current Exp16 path still depends on historical exec-ablation assumptions.",
    ),
    "t6_family_feature_mapping": StageSpec(
        key="t6_family_feature_mapping",
        tech_id="T6",
        order=6,
        title="Family / Feature Mapping",
        objective="Break the late safe-response region into reusable families and probeable features.",
        why_now="This only makes sense after the late safe-response region has been rediscovered.",
        automation_mode="hybrid",
        shieldgemma_policy="Feature naming must stay downstream of behaviorally audited family discovery.",
        human_review_required=True,
        human_review_points=("Check whether feature names remain stable across topics and templates.",),
        experiment_ids=("Exp16", "Exp17"),
        depends_on=("t5_late_safe_response_discovery",),
        runnable_now=False,
        blocked_reason="Late-family discovery is not rebuilt yet, so feature mapping would be premature.",
    ),
    "t7_candidate_quantification": StageSpec(
        key="t7_candidate_quantification",
        tech_id="T7",
        order=7,
        title="Candidate Quantification",
        objective="Quantify only 12B-discovered candidate gates and candidate families.",
        why_now="Quantification before discovery just hardens priors into numbers.",
        automation_mode="mostly_auto",
        shieldgemma_policy="All ranking stays tied to ShieldGemma-audited behavior outcomes.",
        human_review_required=False,
        human_review_points=(),
        experiment_ids=("Exp18",),
        depends_on=("t3_cross_layer_refinement",),
        runnable_now=False,
        blocked_reason="The current quantification script still hardcodes L17 as the main gate candidate.",
    ),
    "t8_minimal_causal_closure": StageSpec(
        key="t8_minimal_causal_closure",
        tech_id="T8",
        order=8,
        title="Minimal Causal Closure",
        objective="Connect the upstream gate story with downstream late-family behavior in the smallest defensible causal chain.",
        why_now="This is the first stage where a 12B mechanism narrative becomes publishable.",
        automation_mode="hybrid",
        shieldgemma_policy="Every closure claim must be backed by ShieldGemma-audited outputs and key-condition sample review.",
        human_review_required=True,
        human_review_points=("Read baseline, gate-only, late-only, and combined samples before accepting any closure story.",),
        experiment_ids=("Exp19",),
        depends_on=("t4_detect_discovery", "t5_late_safe_response_discovery", "t7_candidate_quantification"),
        runnable_now=False,
        blocked_reason="The historical closure script fixes target layers and late layers from 1B.",
    ),
    "t9_robustness": StageSpec(
        key="t9_robustness",
        tech_id="T9",
        order=9,
        title="Robustness",
        objective="Stress-test only already-closed 12B mechanisms across seeds, splits, and topics.",
        why_now="This stage should never run before a 12B-specific closure exists.",
        automation_mode="mostly_auto",
        shieldgemma_policy="ShieldGemma metrics remain the default robustness headline.",
        human_review_required=True,
        human_review_points=("Inspect disagreement and failure cases instead of relabeling everything.",),
        experiment_ids=("Exp12",),
        depends_on=("t8_minimal_causal_closure",),
        runnable_now=False,
        blocked_reason="There is no rebuilt 12B closure to stress-test yet.",
    ),
    "t10_attack_acceptance": StageSpec(
        key="t10_attack_acceptance",
        tech_id="T10",
        order=10,
        title="Attack Acceptance",
        objective="Use attacks only as acceptance tests after 12B mechanism closure is known.",
        why_now="Attack experiments are not allowed to carry discovery.",
        automation_mode="hybrid",
        shieldgemma_policy="ShieldGemma remains the acceptance metric, with manual review on apparent successes.",
        human_review_required=True,
        human_review_points=("Read all apparent attack-success samples and confirm they exploit the intended mechanism.",),
        experiment_ids=("Exp38", "Exp39"),
        depends_on=("t8_minimal_causal_closure",),
        runnable_now=False,
        blocked_reason="No 12B mechanism closure exists yet, so attack acceptance would be premature.",
    ),
    "t11_trace_family_token_detail": StageSpec(
        key="t11_trace_family_token_detail",
        tech_id="T11",
        order=11,
        title="Trace / Family / Token Detail",
        objective="Do token, step, family, and boundary-level mechanism detail only after closure is stable.",
        why_now="Fine-grained tracing is only useful once the main mechanism path has been rebuilt.",
        automation_mode="hybrid",
        shieldgemma_policy="Detailed tracing remains subordinate to behaviorally validated family and closure stories.",
        human_review_required=True,
        human_review_points=("Sample-check traced outputs when a token or family explanation is promoted into the main narrative.",),
        experiment_ids=("Exp17", "Exp20", "Exp21", "Exp22", "Exp23", "Exp24", "Exp25", "Exp26", "Exp27", "Exp28", "Exp29", "Exp30", "Exp31", "Exp32", "Exp33", "Exp34", "Exp35", "Exp36", "Exp37"),
        depends_on=("t5_late_safe_response_discovery", "t8_minimal_causal_closure"),
        runnable_now=False,
        blocked_reason="Mainline gate/detect/late closure has not been rebuilt yet.",
    ),
}


PIPELINE_PRESETS: dict[str, tuple[str, ...]] = {
    "t0_eval_only": ("t0_eval_calibration",),
    "t0_t1_bootstrap": ("t0_eval_calibration", "t1_baseline_diagnosis"),
    "t0_t2_bootstrap": ("t0_eval_calibration", "t1_baseline_diagnosis", "t2_gate_discovery"),
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
            print(f"  - [{stage.tech_id}] {stage.title}")
            for spec in stage.experiment_specs:
                print(f"      * {spec.script}")


def print_stage_table() -> None:
    for stage in sorted(PIPELINE_STAGES.values(), key=lambda item: item.order):
        status = "runnable" if stage.runnable_now else "blocked"
        print(f"{stage.order}. {stage.tech_id} {stage.key} [{status}]")
        print(f"   title: {stage.title}")
        print(f"   objective: {stage.objective}")
        print(f"   experiments: {', '.join(stage.experiment_ids)}")
        print(f"   depends_on: {', '.join(stage.depends_on) if stage.depends_on else 'none'}")
        print(f"   automation: {stage.automation_mode}")
        print(f"   human_review_required: {stage.human_review_required}")
        if stage.blocked_reason:
            print(f"   blocked_reason: {stage.blocked_reason}")


def render_stage_summary(stage_keys: tuple[str, ...]) -> str:
    lines = [
        "# 12B Pipeline Stage Summary",
        "",
        "This run follows the rebuilt 12B-first research chain.",
        "1B experiments are references for ordering only, not for layer priors.",
        "",
    ]
    for stage in sorted((PIPELINE_STAGES[key] for key in stage_keys), key=lambda item: item.order):
        lines.append(f"## {stage.tech_id}: {stage.title}")
        lines.append("")
        lines.append(f"- Key: `{stage.key}`")
        lines.append(f"- Objective: {stage.objective}")
        lines.append(f"- Why now: {stage.why_now}")
        lines.append(f"- Experiment IDs: {', '.join(stage.experiment_ids)}")
        lines.append(f"- Depends on: {', '.join(stage.depends_on) if stage.depends_on else 'none'}")
        lines.append(f"- Automation mode: {stage.automation_mode}")
        lines.append(f"- ShieldGemma policy: {stage.shieldgemma_policy}")
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
