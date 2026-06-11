"""Generate Tic-Tac-Toe boards for the agentic example."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

DEFAULT_COUNT = 128
DEFAULT_SEED = 2026


def generate_records(count: int = DEFAULT_COUNT, *, seed: int = DEFAULT_SEED) -> list[dict]:
    """Generate reproducible legal boards where X is to move."""

    rng = random.Random(seed)
    records: list[dict] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    attempts = 0
    while len(records) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError("could not generate enough unique Tic-Tac-Toe boards")
        board = _random_board(rng)
        key = tuple(tuple(row) for row in board)
        if key in seen or game.is_terminal(board) or game.next_player(board) != "X":
            continue
        seen.add(key)
        records.append({"id": f"generated-{len(records):05d}", "board": board})
    return records


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def _random_board(rng: random.Random) -> game.Board:
    board = [[game.EMPTY, game.EMPTY, game.EMPTY] for _ in range(3)]
    player = "X"
    for _ in range(rng.randint(0, 6)):
        moves = game.legal_moves(board)
        if not moves or game.is_terminal(board):
            break
        board = game.apply_move(board, rng.choice(moves), player)
        player = "O" if player == "X" else "X"
    if game.next_player(board) == "O" and game.legal_moves(board):
        board = game.apply_move(board, rng.choice(game.legal_moves(board)), "O")
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JSONL boards for the Areno Tic-Tac-Toe agentic example.")
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of boards to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")

    records = generate_records(args.count, seed=args.seed)
    if args.output == "-":
        write_jsonl(records, sys.stdout)
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(records, handle)


if __name__ == "__main__":
    main()
