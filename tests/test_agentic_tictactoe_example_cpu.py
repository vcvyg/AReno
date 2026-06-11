from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module(name: str):
    path = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "tictactoe" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"agentic_tictactoe_{name}_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tictactoe_generator_produces_valid_x_to_move_records():
    game = _load_module("game")
    generator = _load_module("dataset_generator")

    records = generator.generate_records(16, seed=7)

    assert len(records) == 16
    for record in records:
        board = game.normalize_board(record["board"])
        assert game.next_player(board) == "X"
        assert not game.is_terminal(board)
        assert game.best_moves(board)


def test_tictactoe_tool_reward_scores_tool_square_only():
    reward = _load_module("reward")
    board = [["X", "X", "."], ["O", ".", "."], ["O", ".", "."]]
    record = SimpleNamespace(
        source_record={"board": board},
        completion="<move>3</move>",
        tool_calls=[{"name": "choose_square", "arguments": {"square": 1}}],
    )

    assert reward.reward_fn(record) == -1.0

    record.tool_calls = [{"name": "choose_square", "arguments": {"square": 3}}]
    assert reward.reward_fn(record) == 1.0


def test_tictactoe_xml_reward_requires_move_tag():
    reward = _load_module("reward_no_tool")
    game = _load_module("game")
    board = [["X", "X", "."], ["O", ".", "."], ["O", ".", "."]]
    record = SimpleNamespace(source_record={"board": board}, completion="3")

    assert reward.reward_fn(record) == -1.0

    record.completion = "<think><move>5</move></think>\n<move>3</move><|im_end|>"
    assert game.parse_xml_move(record.completion) == 3
    assert reward.reward_fn(record) == 1.0
