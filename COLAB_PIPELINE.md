# Colab Pipeline

This is the recommended path for moving the current Gemma-1B mechanism workflow
to a larger model on Colab.

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
```

Optional shared mechanism knobs for larger models:

```python
SCOPE_TOP_K_FAMILY = 8
MIN_GROUP_SIZE = 2
```

## 2. First pass

Do not jump straight to the full stack. Start with:

```bash
!python run_pipeline.py \
  --preset gate_scan \
  --model google/gemma-3-12b-it \
  --hf-token "$HF_TOKEN" \
  --n-train 64 \
  --n-eval 2 \
  --scope-top-k-family $SCOPE_TOP_K_FAMILY \
  --min-group-size $MIN_GROUP_SIZE
```

This checks whether the larger model still has a comparable main gate and which
layers matter.

## 3. Main attack validation

If the gate scan looks real, run:

```bash
!python run_pipeline.py \
  --preset large_model_attack \
  --model google/gemma-3-12b-it \
  --hf-token "$HF_TOKEN" \
  --n-train 64 \
  --n-eval 2 \
  --scope-top-k-family $SCOPE_TOP_K_FAMILY \
  --min-group-size $MIN_GROUP_SIZE
```

This preset runs:

- `exp_01_refusal.py`
- `exp_01b_cross_layer.py`
- `exp_19_l17_l23_late_impact.py`
- `exp_38_whitebox_attack_feasibility.py`
- `exp_39_context_knowledge_bypass.py`
- `analysis/format_attack_reports.py`

## 4. Optional deeper mechanism pass

If you want the family-structure and detect-side picture too:

```bash
!python run_pipeline.py \
  --preset full \
  --model google/gemma-3-12b-it \
  --hf-token "$HF_TOKEN"
```

If you want maximum coverage and are willing to pay the runtime cost:

```bash
!python run_pipeline.py \
  --preset all_experiments \
  --model google/gemma-3-12b-it \
  --hf-token "$HF_TOKEN"
```

## 5. Where results go

Outputs are written to:

```text
results/pipeline_runs/<timestamp>_<model_slug>_<preset>/
```

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

## 6. Practical Colab advice

- Prefer `gate_scan` first. Bigger models may not preserve the exact 1B layer story.
- If the run OOMs, add:
  - `--n-train 64`
  - `--n-eval 2`
  - `--max-new-tokens 96`
- For larger models, do not hide the family knobs:
  - `--scope-top-k-family 8`
  - `--min-group-size 2`
- If you only care about final attack feasibility, rerun just `attack_eval`.
- Copy the entire `results/pipeline_runs/...` directory back to Drive after each major run.

## 7. Interpreting the stages

- `gate_scan`: "Is there still a causal refusal gate?"
- `family_map`: "What downstream safe families depend on that gate?"
- `attack_eval`: "Does the open gate become a stable actionable attack path?"

For large models, this order keeps you from over-reading a single attack result
without first checking whether the gate geometry still matches the 1B regime.

## 8. Transfer caution

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
