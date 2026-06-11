"""Dataset loader for the multi-turn shopping tool-call example."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import make_prompt  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize JSONL shopping rows into prompt-bearing records."""

    rows = default_loader(dataset_path)
    records = []
    for row in rows:
        record = dict(row)
        record["categories"] = list(record["categories"])
        record["required_features_by_category"] = {
            category: list(features) for category, features in record["required_features_by_category"].items()
        }
        record["prompt"] = make_prompt(record)
        records.append(record)
    return records
