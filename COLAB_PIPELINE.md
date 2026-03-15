# Colab Pipeline

This document describes the current `12B`-first mechanism-rebuild workflow.
The key rule remains:

- `1B` is a workflow reference.
- `12B` must rebuild bottom theory from scratch.
- `ShieldGemma` stays the primary safety judge throughout.

## 1. Environment

In a fresh Colab runtime:

```bash
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content
!git clone https://github.com/awaa-col/Surper_GCG_ResearchWorkFlow.git Surper_GCG
%cd /content/Surper_GCG
!pip install -e .
```

If you need gated Hugging Face access:

```python
import os
os.environ["HF_TOKEN"] = "YOUR_TOKEN"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
```

Optional shared knobs:

```python
RUN_NAME = "gemma3_12b_bootstrap"
RESUME = "--resume"  # set to "" for a fresh run
```

## 2. Frontline Starting Point

If you want the first real `12B` frontline tech point, run:

```bash
!python run_pipeline.py \
  --preset baseline_diagnosis \
  --run-name "$RUN_NAME" \
  $RESUME
```

Today this preset runs:

- `exp_00_diagnosis.py`

This is the correct mainline entry if your goal is to start `12B` from scratch
instead of reusing legacy review artifacts.

## 3. Current Theory Bootstrap

If you want the current automated bootstrap for `12B` bottom-theory work, run:

```bash
!python run_pipeline.py \
  --preset theory_rebuild_bootstrap \
  --run-name "$RUN_NAME" \
  $RESUME
```

Today this preset runs:

1. `eval_calibration` (legacy review-pack calibration)
2. `baseline_diagnosis` (real frontline start)

That is the current safe limit for automated `12B` discovery.

## 3C. Next Unlocked Tech Point: Gate Discovery Bootstrap

After `baseline_diagnosis` is complete, the next unlocked mainline step is:

```bash
!python run_pipeline.py \
  --preset gate_discovery_bootstrap \
  --run-name "${RUN_NAME}_gate" \
  $RESUME
```

This preset runs:

1. `baseline_diagnosis`
2. `gate_discovery`

It does not unlock `cross_layer_refinement`, `detect_discovery`, or later
stages yet.

## 3B. Optional Side Quest: Legacy Review-Pack Calibration

If you intentionally want to build a manual review pack from legacy saved
results, run:

```bash
!python run_pipeline.py \
  --preset eval_calibration \
  --run-name "${RUN_NAME}_eval_calibration" \
  $RESUME
```

This is a support tech, not the first real `12B` battle.

## 4. Human Review Policy

The pipeline is intentionally hybrid.

After `eval_calibration`:

- review the exported review pack
- confirm ShieldGemma and manual semantics roughly align
- lock the stage-level audit rubric before moving on

After `baseline_diagnosis`:

- inspect high-risk samples and malformed outputs
- confirm the model has a stable refusal object worth localizing
- write a short baseline memo before attempting gate discovery

Do not auto-advance from baseline into any historical scan wrapper.

## 5. Stage Order

The intended `12B` stage order is:

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

Only stages `0-1` are wired for direct execution right now. The rest remain
blocked until the old scripts are refactored to remove `1B` priors.

## 6. Where Results Go

Outputs are written to:

```text
results/pipeline_runs/<timestamp>_<model_slug>_<preset>/
```

If you pass `--run-name some_fixed_name`, outputs go to:

```text
results/pipeline_runs/some_fixed_name/
```

Each run directory now includes:

- experiment outputs
- `pipeline_manifest.json`
- `pipeline_stage_summary.md`

The stage summary is the quickest way to see:

- what stages were selected
- which human-review checkpoints are required
- which later stages are still blocked

## 7. Practical Advice

- Prefer `baseline_diagnosis` as the first real mainline run.
- Use `eval_calibration` only when you intentionally need a legacy review pack.
- Use `theory_rebuild_bootstrap` when you want both support calibration and the
  real frontline baseline stage in one run.
- If the run OOMs, add:
  - `--n-train 64`
  - `--n-eval 2`
  - `--max-new-tokens 96`
- Keep the same `RUN_NAME` and add `--resume` after Colab restarts.
- Copy the whole `results/pipeline_runs/...` directory back to Drive after each
  major run.

## 8. Transfer Caution

What may transfer from `1B`:

- there may be some upstream refusal gate
- there may be downstream safety-organization effects
- single-direction stories are probably incomplete

What must be rediscovered on `12B`:

- whether a dominant gate exists at all
- where that gate lives
- whether detect and exec still separate
- whether late safe-response structure exists and where
- whether stronger models produce cleaner unsafe execution instead of low-grade
  degeneration

Until those questions are answered, do not treat `L17`, `L23`, `r_exec`, or old
family labels as valid `12B` theory.
