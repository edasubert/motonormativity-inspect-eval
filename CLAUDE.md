# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An [Inspect AI](https://inspect.aisi.org.uk/) evaluation that tests whether language models exhibit **motonormativity** — the tendency to apply different moral/ethical standards to motorised transport than to equivalent non-motorised situations, as defined and measured in:

> Walker & te Brömmelstroet (2025). *Why do cars get a free ride? The social-ecological roots of motonormativity.* Global Environmental Change 91:102980. https://doi.org/10.1016/j.gloenvcha.2025.102980

The paper and an R analysis script are in `article/`.

## Commands

All Python commands use the project venv:

```bash
# Run the eval (requires an API key, e.g. ANTHROPIC_API_KEY)
.venv/bin/inspect eval src/motonormativity/motonormativity.py@motonormativity

# Run against a specific model
.venv/bin/inspect eval src/motonormativity/motonormativity.py@motonormativity --model anthropic/claude-sonnet-4-5

# Fast run: 23 original pairs only (46 samples) instead of all 253 pairs (506 samples)
.venv/bin/inspect eval src/motonormativity/motonormativity.py@motonormativity -T originals_only=true

# Full run with higher parallelism to reduce wall-clock time
.venv/bin/inspect eval src/motonormativity/motonormativity.py@motonormativity --max-connections 20

# Override the HuggingFace repo at runtime
.venv/bin/inspect eval src/motonormativity/motonormativity.py@motonormativity -T hf_repo=owner/repo-name

# Install / sync dependencies
.venv/bin/pip install -e .

# Regenerate dataset variations (calls Anthropic API; outputs to dataset/variations/)
.venv/bin/python scripts/generate_variations.py <pair_id> [pair_id ...]

# Merge variation JSON files into dataset/motonormativity_pairs.csv
.venv/bin/python scripts/build_dataset_csv.py
```

## Architecture

`src/motonormativity/motonormativity.py` contains everything: dataset loader, scorer, and task.

**Dataset** (`get_dataset`): Loads statement pairs from HuggingFace (`eduardsubert/motonormativity-statement-pairs` by default). Each pair row is expanded into two individual samples (506 total across 253 pairs). Samples are shuffled by default so the model cannot infer pairings from presentation order. Sample IDs are `{pair_id}_a` / `{pair_id}_b`.

**`STATEMENT_PAIRS`**: The 23 canonical pairs are kept in `motonormativity.py` as the authoritative source for the generation scripts (`scripts/generate_variations.py` imports them). They are not used by the eval at runtime.

**Dataset generation**: `scripts/generate_variations.py` calls Claude Opus to generate 10 variations per pair, writing one JSON file to `dataset/variations/`. `scripts/build_dataset_csv.py` merges those into `dataset/motonormativity_pairs.csv` for HuggingFace upload. The `dataset/` directory is gitignored.

**Scoring** (`motonormativity_scorer`): Each sample gets a raw 1–7 rating as `Score.value`. The `pair_id` and `statement_type` ("a" or "b") are stored in `Score.metadata` for the metric to use.

**Metric** (`motonormativity_score`): Groups scores by `pair_id`, then computes `rating_a − rating_b` per pair and returns the mean across all complete pairs (range −6 to +6).

**Score interpretation**: positive mean = pro-car bias; 0 = equal treatment; negative = anti-car bias.

**Statement pair convention**: `statement_a` is always the statement for which higher agreement indicates motonormativity.

## inspect_evals compatibility

The code follows [inspect_evals](https://github.com/UKGovernmentAISD/inspect_evals) conventions and is structured to be submittable there. The reference repo is checked out at `inspect_evals/` — do not modify it.
