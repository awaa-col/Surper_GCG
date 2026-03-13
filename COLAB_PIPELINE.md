# Colab Pipeline

This file documents only the entrypoints that are still allowed for mechanism
work. Old `L17/L23`-anchored scan presets were removed and should not be run.

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

## 3. What To Do Next

After `mechanism_discovery_foundation`, stop. Do not continue into any old scan
wrapper that hardcodes historical layer assumptions.

There is no public scan wrapper right now. That is intentional.

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

The same run directory contains only the artifacts produced by the preset you
explicitly ran.

## 5. Practical Advice

- Prefer `mechanism_discovery_foundation` first.
- If the run OOMs, add:
  - `--n-train 64`
  - `--n-eval 2`
  - `--max-new-tokens 96`
- For larger models, do not hide the family knobs:
  - `--scope-top-k-family 8`
  - `--min-group-size 2`
- For Colab restarts, keep the same `RUN_NAME` and add `--resume`.
- Copy the entire `results/pipeline_runs/...` directory back to Drive after each major run.

## 6. Interpreting the Stages

- `eval_calibration`: "Are our labels and review artifacts stable enough to trust later results?"
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
