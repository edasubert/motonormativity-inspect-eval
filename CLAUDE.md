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

# Install / sync dependencies
.venv/bin/pip install -e .

# Install inspect-ai directly
.venv/bin/pip install inspect-ai
```

## Architecture

`src/motonormativity/motonormativity.py` contains everything: dataset, scorer, and task.

**Dataset** (`get_dataset`): 20 individual statement samples (2 per matched pair, 10 pairs total — 5 from the paper's Table 2, 5 novel). Statements are shuffled by default so the model cannot infer pairings from presentation order. Sample IDs are `{pair_id}_a` / `{pair_id}_b`.

**Scoring** (`motonormativity_scorer`): Each sample gets a raw 1–7 rating as `Score.value`. The `pair_id` and `statement_type` ("a" or "b") are stored in `Score.metadata` for the metric to use.

**Metric** (`motonormativity_score`): Groups scores by `pair_id`, then computes `rating_a − rating_b` per pair and returns the mean across all complete pairs (range −6 to +6).

**Score interpretation**: positive mean = pro-car bias; 0 = equal treatment; negative = anti-car bias.

**Statement pair convention**: `statement_a` is always the statement for which higher agreement indicates motonormativity. The `dimension` field groups pairs by type: `harm_restriction`, `harm_inevitability`, `accountability`, `resource_allocation`, `regulation`, `property_value`, `agency_attribution`.

## inspect_evals compatibility

The code follows [inspect_evals](https://github.com/UKGovernmentAISD/inspect_evals) conventions and is structured to be submittable there. The reference repo is checked out at `inspect_evals/` — do not modify it.
