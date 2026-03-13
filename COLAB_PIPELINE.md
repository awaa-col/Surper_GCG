# Colab Pipeline

This file now only documents the presets that are still valid entry points for
mechanism work.

If your goal is accurate `12B` mechanism discovery, do not start with attack
evaluation or old transfer-validation stacks.

## 1. Environment

In a fresh Colab runtime:

```bash
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content
!git clone YOUR_REPO_URL Surper_GCG
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

Optional shared mechanism knobs for larger models:

```python
RUN_NAME = "gemma3_12b_main"
RESUME = "--resume"  # set to "" for a fresh run
SCOPE_TOP_K_FAMILY = 8
MIN_GROUP_SIZE = 2
```

## 2. Accurate Starting Point

Start with evaluation calibration first:

```bash
!python run_pipeline.py \
  --preset mechanism_discovery_foundation \
  --run-name "$RUN_NAME" \
  $RESUME
```

Today this preset only runs:

- `exp_11_review_pack.py`

That is intentional. It is the only stage that currently enters the `12B`
mechanism-discovery mainline without carrying the old `1B` layer prior.

## 3. Scan Pipeline

If you need the current larger-model scan preset before the old `1B`-anchored
scan logic is refactored, run:

```bash
!python run_pipeline.py \
  --preset mechanism_scan_legacy \
  --run-name "$RUN_NAME" \
  $RESUME \
  --model google/gemma-3-12b-it \
  --hf-token "$HF_TOKEN" \
  --n-train 64 \
  --n-eval 2 \
  --scope-top-k-family $SCOPE_TOP_K_FAMILY \
  --min-group-size $MIN_GROUP_SIZE
```

This preset runs:

- `exp_00_diagnosis.py`
- `exp_01_refusal.py`
- `exp_01b_cross_layer.py`
- `exp_16_safe_response_dictionary.py`
- `exp_17_gemma_scope_feature_probe.py`
- `exp_19_l17_l23_late_impact.py`

This is the current scan pipeline for data collection. It is still provisional:
the logic is useful for gathering `12B` scan evidence, but it is not yet the
final prior-free `12B` mechanism-discovery mainline.

If you want maximum coverage and are willing to pay the runtime cost:

```bash
!python run_pipeline.py \
  --preset all_experiments \
  --model google/gemma-3-12b-it \
  --hf-token "$HF_TOKEN"
```

## 4. Where Results Go

Outputs are written to:

```text
results/pipeline_runs/<timestamp>_<model_slug>_<preset>/
```

If you pass `--run-name some_fixed_name`, outputs go to:

```text
results/pipeline_runs/some_fixed_name/
```

Then `--resume` will skip stages whose output files already exist in that run directory.

The manifest file records:

- model name
- preset
- exact commands
- output file paths
- extra args

The same run directory also contains:

- `exp38_summary.md`
- `exp39_generations.md`

For `all_experiments`, many scripts intentionally keep writing to their native
default locations under `results/`, because later experiments depend on
those default file names.

## 5. Practical Advice

- Prefer `mechanism_discovery_foundation` first.
- If the run OOMs, add:
  - `--n-train 64`
  - `--n-eval 2`
  - `--max-new-tokens 96`
- For larger models, do not hide the family knobs:
  - `--scope-top-k-family 8`
  - `--min-group-size 2`
- If you only care about final attack feasibility, rerun just `attack_eval`.
- For Colab restarts, keep the same `RUN_NAME` and add `--resume`.
- Copy the entire `results/pipeline_runs/...` directory back to Drive after each major run.

## 6. Interpreting the Stages

- `eval_calibration`: "Are our labels and review artifacts stable enough to trust later results?"
- `mechanism_scan_legacy`: "Collect provisional larger-model scan data with the old scan logic."
- `family_map`: "What downstream safe families depend on that gate?"

For accurate `12B` mechanism work, the intended order is:

1. `eval_calibration`
2. refactor the old gate/detect/late scripts to remove `1B` layer priors
3. only then rerun discovery and later attack acceptance if needed

## 7. Transfer Caution

Your 1B findings are a strong hypothesis, not a guaranteed transplant.

What is likely to transfer:

- there is some upstream refusal gate,
- there are downstream safety-family organization effects,
- single-direction stories are probably incomplete.

What must be re-validated on larger models:

- the exact key layer or layer pair,
- whether `r_exec` stays the dominant causal handle,
- whether knowledge injection still separates "capacity limit" from "safety suppression",
- whether stronger models replace 1B's low-fidelity unsafe output with cleaner unsafe execution.

Two specific scripts remain blocked on this point:

- `exp_18_l17_vector_quantification.py`
- `exp_19_l17_l23_late_impact.py`

They must be refactored before they can re-enter the accurate `12B` mainline.
