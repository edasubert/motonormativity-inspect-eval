#!/usr/bin/env python3
"""Merge all variation JSON files into dataset/motonormativity_pairs.csv.

Run after generate_variations.py has produced all dataset/variations/*.json files.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from motonormativity.motonormativity import STATEMENT_PAIRS


def main() -> None:
    var_dir = Path(__file__).parent.parent / "dataset" / "variations"
    out_path = Path(__file__).parent.parent / "dataset" / "motonormativity_pairs.csv"

    rows = []
    missing = []

    for pair in STATEMENT_PAIRS:
        rows.append({
            "id": pair["id"],
            "base_id": pair["id"],
            "variation": 0,
            "source": pair["source"],
            "statement_a": pair["statement_a"],
            "statement_b": pair["statement_b"],
        })

        var_file = var_dir / f"{pair['id']}.json"
        if not var_file.exists():
            missing.append(pair["id"])
            continue

        data = json.loads(var_file.read_text(encoding="utf-8"))
        for i, var in enumerate(data["variations"], 1):
            rows.append({
                "id": f"{pair['id']}_v{i:02d}",
                "base_id": pair["id"],
                "variation": i,
                "source": pair["source"],
                "statement_a": var["statement_a"],
                "statement_b": var["statement_b"],
            })

    if missing:
        print(f"WARNING: Missing variation files for: {missing}", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "base_id", "variation", "source", "statement_a", "statement_b"],
        )
        writer.writeheader()
        writer.writerows(rows)

    pairs_with_variations = len(STATEMENT_PAIRS) - len(missing)
    print(f"Written {len(rows)} rows to {out_path}")
    print(f"  {len(STATEMENT_PAIRS)} originals + {pairs_with_variations * 10} variations")
    if missing:
        print(f"  {len(missing)} pairs missing variations (originals still included)")


if __name__ == "__main__":
    main()
