# Super GCG POC

This directory now has a pipeline-style entrypoint for running the current
mechanism experiments locally or on Colab.

## Quick start

Install the package from this directory:

```bash
pip install -e .
```

Run the pipeline entrypoint:

```bash
python run_pipeline.py --help
```

List available presets:

```bash
python run_pipeline.py --list-presets
```

Run the recommended large-model evaluation path:

```bash
python run_pipeline.py \
  --preset large_model_attack \
  --model google/gemma-3-12b-it \
  --hf-token YOUR_TOKEN \
  --n-train 64 \
  --n-eval 2
```

If you prefer a ready-to-run Colab notebook, open:

```text
Surper_GCG_Colab.ipynb
```

Dry-run the commands without executing:

```bash
python run_pipeline.py \
  --preset large_model_attack \
  --model google/gemma-3-12b-it \
  --dry-run
```

## Recommended preset order

- `gate_scan`: find the main refusal gate and cross-layer transfer pattern.
- `family_map`: map late-family and detect-side interactions.
- `attack_eval`: validate white-box attack feasibility and knowledge bypass.
- `analysis_attack_eval`: turn `exp38` and `exp39` JSON outputs into Markdown reports.
- `large_model_attack`: practical Colab preset for bigger models.
- `full`: run all of the above in sequence.
- `all_experiments`: run every discovered `exp_*.py` script in numeric order.

## Outputs

Pipeline outputs are grouped under:

```text
results/pipeline_runs/<run_name>/
```

Each experiment result is written into that run directory. The pipeline also
stores a `pipeline_manifest.json` with the exact commands used.

For attack-eval runs, the same directory also contains:

- `exp38_summary.md`
- `exp39_generations.md`

For `all_experiments`, the runner preserves each script's native default output
paths under `results/` so that cross-experiment dependencies keep working.

## Notes for larger models

- Start with `gate_scan` before reusing the Gemma-1B assumptions on a bigger model.
- `exp_38_whitebox_attack_feasibility.py` and
  `exp_39_context_knowledge_bypass.py` are the most useful end-to-end attack
  checks once the gate hypothesis looks real.
- Do not assume the 1B layer story transfers unchanged. Re-check the main gate
  layer, the important layer pairs, and whether the same interventions stay
  coherent instead of only increasing risk.
- If Colab VRAM is tight, reduce `--n-train`, `--n-eval`, and
  `--max-new-tokens` through the pipeline-level `--n-train`, `--n-eval`, and
  `--max-new-tokens` flags.
- If you want maximum coverage instead of a curated path, run:

```bash
python run_pipeline.py \
  --preset all_experiments \
  --model google/gemma-3-12b-it \
  --hf-token YOUR_TOKEN
```

See [COLAB_PIPELINE.md](/g:/Surper_GCG/poc/COLAB_PIPELINE.md) for a concrete
Colab workflow.
