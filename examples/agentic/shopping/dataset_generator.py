"""Generate JSONL shopping kit tasks for the multi-turn agentic example."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TASKS = [
    {
        "kit_name": "rain commute",
        "categories": ["jacket", "bottle"],
        "budget": 140,
        "required_features_by_category": {
            "jacket": ["waterproof", "packable"],
            "bottle": ["insulated", "leakproof"],
        },
    },
    {
        "kit_name": "trail day",
        "categories": ["shoes", "bottle"],
        "budget": 170,
        "required_features_by_category": {
            "shoes": ["trail", "water-resistant"],
            "bottle": ["collapsible", "lightweight"],
        },
    },
    {
        "kit_name": "cold city",
        "categories": ["jacket", "shoes"],
        "budget": 230,
        "required_features_by_category": {
            "jacket": ["windproof", "warm"],
            "shoes": ["casual", "water-resistant"],
        },
    },
    {
        "kit_name": "full travel",
        "categories": ["jacket", "shoes", "bottle"],
        "budget": 260,
        "required_features_by_category": {
            "jacket": ["waterproof", "packable"],
            "shoes": ["casual", "water-resistant"],
            "bottle": ["collapsible", "lightweight"],
        },
    },
]


def generate_records(count: int, *, seed: int = 2026) -> list[dict]:
    """Generate deterministic shopping records."""

    rng = random.Random(seed)
    records = []
    for idx in range(count):
        task = dict(rng.choice(TASKS))
        task["categories"] = list(task["categories"])
        task["required_features_by_category"] = {
            category: list(features) for category, features in task["required_features_by_category"].items()
        }
        task["id"] = idx
        records.append(task)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL tasks for the Areno shopping agentic example.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--count", type=int, default=256, help="Number of records.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    args = parser.parse_args()

    records = generate_records(args.count, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
