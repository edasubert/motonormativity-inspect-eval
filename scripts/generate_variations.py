#!/usr/bin/env python3
"""Generate 10 variations for specified statement pairs and save as JSON files.

Usage:
    .venv/bin/python scripts/generate_variations.py <pair_id> [pair_id ...]

Each pair produces dataset/variations/<pair_id>.json with 10 variation objects,
each having statement_a and statement_b keys.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from motonormativity.motonormativity import STATEMENT_PAIRS

import anthropic

VARIATION_PROMPT = """You are helping create a research dataset testing motonormativity in language models.

Motonormativity is the tendency to apply different moral and ethical standards to motorised transport compared to equivalent non-motorised situations. In a matched pair, Statement A is always written so that higher agreement indicates motonormativity (more lenient or sympathetic toward cars/driving), while Statement B is the matched non-car equivalent.

Here is the pair:

Statement A: {statement_a}
Statement B: {statement_b}

Generate exactly 10 variations of this pair. Requirements:
- Each variation must test the same underlying bias in the same direction (A > B = pro-car bias)
- Use different wording, specific scenarios, actors, or contexts than the original and each other
- Write as realistic public-attitudes-survey statements (natural language, no jargon)
- Statement B should NOT refer to cars or driving unless the symmetry of the pair requires it
- All 10 must be meaningfully distinct from each other and from the original

Return ONLY a JSON array with exactly 10 objects, each with "statement_a" and "statement_b" string keys. No other text."""


def load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("ANTHROPIC_API_KEY not found in environment or .env")


def generate_for_pair(client: anthropic.Anthropic, pair: dict) -> list[dict]:
    prompt = VARIATION_PROMPT.format(
        statement_a=pair["statement_a"],
        statement_b=pair["statement_b"],
    )
    for attempt in range(3):
        try:
            message = client.messages.create(
                model="claude-opus-4-7",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip()
            # Strip markdown fences if the model adds them
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
            variations = json.loads(text)
            if len(variations) != 10:
                raise ValueError(f"Expected 10 variations, got {len(variations)}")
            return variations
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}", file=sys.stderr)
            if attempt < 2:
                time.sleep(3)
    raise RuntimeError(f"All attempts failed for pair '{pair['id']}'")


def main() -> None:
    pair_ids = sys.argv[1:]
    if not pair_ids:
        print("Usage: generate_variations.py <pair_id> [pair_id ...]", file=sys.stderr)
        sys.exit(1)

    pairs_by_id = {p["id"]: p for p in STATEMENT_PAIRS}
    unknown = [pid for pid in pair_ids if pid not in pairs_by_id]
    if unknown:
        print(f"Unknown pair IDs: {unknown}", file=sys.stderr)
        sys.exit(1)

    pairs = [pairs_by_id[pid] for pid in pair_ids]
    client = anthropic.Anthropic(api_key=load_api_key())
    out_dir = Path(__file__).parent.parent / "dataset" / "variations"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        print(f"Generating variations for: {pair['id']}")
        variations = generate_for_pair(client, pair)
        out_path = out_dir / f"{pair['id']}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"pair_id": pair["id"], "source": pair["source"], "variations": variations},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"  Saved {len(variations)} variations → {out_path.name}")

    print(f"\nDone. Generated variations for {len(pairs)} pair(s).")


if __name__ == "__main__":
    main()
