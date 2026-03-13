# Super GCG POC

This directory has a pipeline-style entrypoint for running the current
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

Run the accurate starting point:

```bash
python run_pipeline.py \
  --preset mechanism_discovery_foundation
```

If you prefer a ready-to-run Colab notebook, open:

```text
Surper_GCG_Colab.ipynb
```

Dry-run the commands without executing:

```bash
python run_pipeline.py \
  --preset mechanism_discovery_foundation \
  --dry-run
```

## Available presets

- `baseline_diagnosis`: run the baseline diagnosis entrypoint.
- `eval_calibration`: export review-pack artifacts.
- `mechanism_discovery_foundation`: current safe mainline entrypoint.

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

## Notes

- Start with `mechanism_discovery_foundation`.
- Do not use the pipeline to run old `L17/L23`-anchored scan logic. Those entry
  points were removed on purpose.
- If Colab VRAM is tight, reduce `--n-train`, `--n-eval`, and
  `--max-new-tokens` through the pipeline-level `--n-train`, `--n-eval`, and
  `--max-new-tokens` flags.

See [COLAB_PIPELINE.md](/g:/Surper_GCG/poc/COLAB_PIPELINE.md) for a concrete
Colab workflow.
