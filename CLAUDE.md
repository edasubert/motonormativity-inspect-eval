# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An [Inspect AI](https://inspect.aisi.org.uk/) evaluation that tests whether language models exhibit **motonormativity** — the tendency to apply different moral/ethical standards to motorised transport than to equivalent non-motorised situations, as defined and measured in:

> Walker & te Brömmelstroet (2025). *Why do cars get a free ride? The social-ecological roots of motonormativity.* Global Environmental Change 91:102980. https://doi.org/10.1016/j.gloenvcha.2025.102980

The eval presents each statement of a matched pair individually (random order, without showing the pair), records a 1–7 agreement rating, and reports **per-construct** results benchmarked against the human values from the paper.

---

## Architecture

`src/motonormativity/motonormativity.py` contains the dataset loader, scorer, metric, and task. The human reference values are loaded from the HuggingFace dataset at task construction time.

### `_load_human_reference(hf_repo)`

Loads the `human_reference` config from HuggingFace. Returns `{category: {"mean": float, "sd": float}}` — the published `(rating_a − rating_b)` values from Walker & te Brömmelstroet (2025), Table 2. Cached per process so repeated task construction is free.

### `get_dataset`

Loads the `statements` config from HuggingFace and expands each pair into two individually-presented samples (`{category}_{variation}_{a|b}`). Carries `category`, `pair_id`, `statement_type`, `source`, and `statement` in `Sample.metadata`. Shuffled by default. `originals_only=True` filters to `variation == 0` (5 literature originals → 10 samples).

Dataset: 5 constructs × 11 variations = 55 pairs / 110 samples (full); 5 pairs / 10 samples (`originals_only`).

### `_extract_rating`

Robust 1–7 parser. Steps in order:

1. Strip scale-range boilerplate ("on a scale of 1 to 7").
2. Reject ambiguous range answers ("5–6", "5 or 6") → `None`.
3. Return the digit after an explicit label (`Answer:`/`Rating:`/`Final:`/etc.).
4. Otherwise return the **last** standalone 1–7 integer (blocks `10`, `2025`, `3.5`).
5. Return `None` on failure; scorer maps `None` to `nan`.

### `motonormativity_by_construct(human_reference)`

`@metric` that receives all `SampleScore` objects and returns a flat `dict[str, float]` (Inspect renders each key as a named metric value). The `human_reference` argument is mandatory — it must be pre-loaded before API calls begin.

Grouping: `category → pair_id → epoch → {a, b}`. Computes per-construct:

| Key | Description |
|-----|-------------|
| `{cat}_model_mean` | mean(rating_a − rating_b) over all paraphrases and epochs |
| `{cat}_sd_between_paraphrase` | SD across paraphrases of each paraphrase's epoch-averaged diff (wording sensitivity; NaN with 1 paraphrase) |
| `{cat}_sd_within_paraphrase` | mean across paraphrases of the SD across epochs for a fixed paraphrase (run-to-run noise; NaN without multi-epoch data) |
| `{cat}_se` | `sd_between_paraphrase / sqrt(n_paraphrases)` (NaN with 1 paraphrase) |
| `{cat}_n_paraphrases` | number of paraphrases with complete pairs |
| `{cat}_n_obs` | total complete pair observations (paraphrases × epochs) |
| `{cat}_human_mean` | reference `(a − b)` from the paper (omitted if construct not in reference) |
| `{cat}_human_sd` | reference SD from the paper (omitted if construct not in reference) |
| `parse_failure_rate` | fraction of samples where the rating could not be parsed |

### `motonormativity_scorer(human_reference)`

Factory that returns a `Scorer`. Accepts the pre-loaded `human_reference` dict and attaches the corresponding `motonormativity_by_construct` metric instance to the scorer via `@scorer(metrics=[...])`. This is the correct pattern for Inspect: metrics on `Task(metrics=[...])` are not applied to named scorers — they must be baked in at the scorer level.

Records the raw 1–7 rating as `Score.value` (`nan` if unparsed). Writes `category`, `pair_id`, `statement_type`, and `epoch` to `Score.metadata`.

### `motonormativity` (task)

Calls `_load_human_reference` once at construction, then passes it to `motonormativity_scorer`. Epoch reduction: `no_epoch_reduction=True` passes `reducer=[]` to `Epochs`, which delivers every individual draw to the metric; `False` (default) uses Inspect's mean reducer so the metric sees one score per sample.

---

## Scoring convention

- Per construct, `model (a − b)` is compared to the human `(a − b)`. Distance from 0 = how unequal the model's standards are.
- There is **no** global "positive = pro-car" rule. `statement_a` is the **non-car** side for `fumes`/`noise` and the **car** side for `fatalism`/`responsibility`/`subsidy`. Mixed signs in the human reference are correct and expected.
- Aggregate at the **construct level**. Paraphrases are within-construct replicates, not independent observations.

Human reference values (Walker & te Brömmelstroet 2025, Table 2; pooled N=2035, 7-pt within-subject):

| Construct | human_mean | human_sd |
|-----------|-----------|---------|
| fumes | +2.15 | 1.87 |
| noise | +0.39 | 1.60 |
| fatalism | −0.43 | 1.53 |
| responsibility | −0.29 | 1.30 |
| subsidy | −0.91 | 1.94 |

---

## Commands

```bash
# Run the eval (requires ANTHROPIC_API_KEY or similar)
.venv/bin/inspect eval src/motonormativity/motonormativity.py@motonormativity

# Specific model
.venv/bin/inspect eval src/motonormativity/motonormativity.py@motonormativity --model anthropic/claude-sonnet-4-5

# Fast run: 5 literature originals only (10 samples)
.venv/bin/inspect eval src/motonormativity/motonormativity.py@motonormativity -T originals_only=true

# Multiple epochs with variance decomposition (use non-zero temperature)
.venv/bin/inspect eval src/motonormativity/motonormativity.py@motonormativity \
    --epochs 5 -T no_epoch_reduction=true

# Override the HuggingFace repo
.venv/bin/inspect eval src/motonormativity/motonormativity.py@motonormativity -T hf_repo=owner/repo-name

# Tests
.venv/bin/pytest tests/
.venv/bin/pytest tests/ --cov=motonormativity --cov-report=term-missing

# Install / sync
.venv/bin/pip install -e .
```

### Epochs and variance decomposition

With `--epochs N -T no_epoch_reduction=true` (and temperature > 0), Inspect passes all N×110 individual scores to the metric. The metric then decomposes spread into:

- **`sd_between_paraphrase`** — wording sensitivity (each paraphrase averaged over epochs first, so decoding noise is removed). This is the per-construct uncertainty for error bars.
- **`sd_within_paraphrase`** — run-to-run noise (SD across epochs for a fixed paraphrase, averaged over paraphrases). Diagnostic only; NaN in single-epoch runs.

Without `no_epoch_reduction` (or with temperature 0), Inspect mean-reduces epochs per sample and the metric sees one score per sample. `sd_within_paraphrase` is NaN and `sd_between_paraphrase` reflects wording sensitivity only.

---

## Dataset generation scripts

`scripts/` builds the dataset offline (gitignored output, uploaded to HF separately). Produces the 5-construct, paraphrase-only set with two configs: `statements` (55 pairs) and `human_reference` (5 rows).

## inspect_evals compatibility

The code follows [inspect_evals](https://github.com/UKGovernmentAISD/inspect_evals) conventions and is structured to be submittable there. If the reference repo is checked out at `inspect_evals/`, do not modify it.
