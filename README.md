# Super GCG POC

This directory contains the pipeline entrypoint for the current `12B`-first
mechanism-rebuild workflow.

Core policy:

- `1B` experiments are workflow references, not `12B` priors.
- `12B` bottom theory must be rebuilt from scratch.
- `ShieldGemma` is the default safety judge for every pipeline stage.

Workflow definitions live under [pipeline/README.md](/g:/Surper_GCG/poc/pipeline/README.md).

## Quick Start

Install the package from this directory:

```bash
pip install -e .
```

Run the pipeline entrypoint help:

```bash
python run_pipeline.py --help
```

List available presets:

```bash
python run_pipeline.py --list-presets
```

List the full stage catalog, including blocked stages and human-review
checkpoints:

```bash
python run_pipeline.py --list-stages
```

Run the frontline starting point:

```bash
python run_pipeline.py \
  --preset baseline_diagnosis
```

Run the current theory bootstrap:

```bash
python run_pipeline.py \
  --preset theory_rebuild_bootstrap
```

Dry-run without executing:

```bash
python run_pipeline.py \
  --preset theory_rebuild_bootstrap \
  --dry-run
```

If you prefer a ready-to-run Colab notebook, open:

```text
Surper_GCG_Colab.ipynb
```

## Available Presets

- `eval_calibration`: export the review-pack artifacts for manual audit.
- `baseline_diagnosis`: run the baseline diagnosis entrypoint.
- `mechanism_discovery_foundation`: current minimal safe mainline; today it
  only runs `eval_calibration` as a support calibration step.
- `theory_rebuild_bootstrap`: current safe automated bootstrap for `12B`
  bottom-theory work; runs `eval_calibration` then `baseline_diagnosis`.

## Stage Model

The pipeline now tracks the intended `12B` mechanism-rebuild order explicitly:

1. `eval_calibration`
2. `baseline_diagnosis`
3. `gate_discovery`
4. `cross_layer_refinement`
5. `detect_discovery`
6. `late_safe_response_discovery`
7. `candidate_quantification`
8. `minimal_causal_closure`
9. `robustness`
10. `attack_acceptance`

Only the first two stages are currently runnable. The later stages remain
blocked until their old `1B` priors are removed from the underlying scripts.

## Outputs

Pipeline outputs are grouped under:

```text
results/pipeline_runs/<run_name>/
```

Each run directory contains:

- experiment outputs
- `pipeline_manifest.json`
- `pipeline_stage_summary.md`

The stage summary records:

- stage objectives
- why the stage exists now
- ShieldGemma policy
- required human-review checkpoints
- blocked-stage notes for the selected preset

## Notes

- Start with `baseline_diagnosis` for the first real `12B` mainline run.
- Use `mechanism_discovery_foundation` only if you explicitly want the legacy
  review-pack calibration step.
- Use `theory_rebuild_bootstrap` when you want the current automated `12B`
  bottom-theory bootstrap.
- Do not run old `L17/L23`-anchored scan wrappers as if they were valid `12B`
  discovery stages.
- The pipeline is intentionally hybrid: it automates generation, ShieldGemma
  audit, ranking, and artifact export, but several stages still require human
  review before advancing.
- If Colab VRAM is tight, reduce `--n-train`, `--n-eval`, and
  `--max-new-tokens` through the shared pipeline flags.

See [COLAB_PIPELINE.md](/g:/Surper_GCG/poc/COLAB_PIPELINE.md) for the Colab
workflow.
