# Super GCG Research Workflow

This directory contains the rebuilt canonical pipeline entrypoint for the `12B`-first mechanism-discovery workflow.

Core policy:

- `1B` experiments are workflow references, not `12B` priors.
- `12B` bottom theory must be rebuilt from scratch.
- `ShieldGemma` is the default safety judge for every pipeline stage.

Workflow definitions live under [pipeline/README.md](/g:/Surper_GCG/pipeline/README.md).

## Quick Start

Install the package from this directory:

```bash
pip install -e .
```

Show help:

```bash
python run_pipeline.py --help
```

List presets:

```bash
python run_pipeline.py --list-presets
```

List the full stage catalog:

```bash
python run_pipeline.py --list-stages
```

Run the first clean mainline bootstrap:

```bash
python run_pipeline.py \
  --preset t0_t1_bootstrap
```

Run the full currently-runnable chain:

```bash
python run_pipeline.py \
  --preset t0_t2_bootstrap
```

Dry-run without executing:

```bash
python run_pipeline.py \
  --preset t0_t2_bootstrap \
  --dry-run
```

If you prefer a Colab notebook, open:

```text
Surper_GCG_Colab.ipynb
```

## Available Presets

- `t0_eval_only`: run only the evaluation-calibration support tech.
- `t0_t1_bootstrap`: run `T0 -> T1`.
- `t0_t2_bootstrap`: run `T0 -> T1 -> T2`.

## Stage Model

The pipeline now tracks the intended `12B` stage order explicitly:

1. `T0 eval_calibration`
2. `T1 baseline_diagnosis`
3. `T2 gate_discovery`
4. `T3 cross_layer_refinement`
5. `T4 detect_discovery`
6. `T5 late_safe_response_discovery`
7. `T6 family_feature_mapping`
8. `T7 candidate_quantification`
9. `T8 minimal_causal_closure`
10. `T9 robustness`
11. `T10 attack_acceptance`
12. `T11 trace_family_token_detail`

Only `T0-T2` are currently wired for direct execution. The later stages remain blocked until their old `1B` priors are removed from the underlying scripts.

## Outputs

Pipeline outputs are grouped under:

```text
results/pipeline_runs/<run_name>/
```

Each run directory contains:

- experiment outputs
- `pipeline_manifest.json`
- `pipeline_stage_summary.md`

## Notes

- Start with `t0_t1_bootstrap` for the first clean `12B` mainline run.
- Use `t0_eval_only` only when you intentionally want the review-pack calibration step.
- Use `t0_t2_bootstrap` when you want the current full runnable chain.
- Treat `T2 gate_discovery` as a provisional manual-review shortlist, not a final gate claim.
- Do not run old `L17/L23`-anchored scan wrappers as if they were valid `12B` discovery stages.
- The pipeline is intentionally hybrid: it automates generation, ShieldGemma audit, ranking, and artifact export, but several stages still require human review before advancing.
- If Colab VRAM is tight, reduce `--n-train`, `--n-eval`, and `--max-new-tokens`.

See [COLAB_PIPELINE.md](/g:/Surper_GCG/COLAB_PIPELINE.md) for the Colab workflow.
