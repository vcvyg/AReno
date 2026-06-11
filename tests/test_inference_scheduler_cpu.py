from __future__ import annotations

from types import SimpleNamespace

import torch

import areno.engine.inference as inference_mod
from areno.engine.api import _chunk_prompts_for_prefill_budget
from areno.engine.data import SamplingParams
from areno.engine.data.rollout_state import InferenceBatchState
from areno.engine.inference import InferenceManager
from tests.helpers import PatchedContext


class _FakeInferenceManager(InferenceManager):
    """CPU-only manager that exercises rollout scheduling without a model."""

    def __init__(self):
        super().__init__(SimpleNamespace())
        self.device = torch.device("cpu")

    def _infer_next_token_tensor(self, payload):
        count = int(payload.sample_indices.numel())
        return torch.ones(count, dtype=torch.long), torch.zeros(count, dtype=torch.float32)

    def _infer_decode_next_token_tensor(
        self,
        next_tokens,
        position_ids,
        cache_seqlens,
        block_table,
        active_count,
        sampling_params,
        sample_generator,
        *,
        sample_step,
        eos_token_id,
    ):
        del position_ids, cache_seqlens, block_table, active_count, sampling_params, sample_generator, eos_token_id
        return next_tokens + 1, torch.zeros_like(next_tokens, dtype=torch.float32) - float(sample_step)

def test_no_sync_rollout_continues_pending_prompts_beyond_running_slots():
    """A single rollout call should drain pending rows while respecting slot limits."""

    manager = _FakeInferenceManager()
    state = InferenceBatchState(
        prompts=[[10], [11], [12], [13]],
        max_new_tokens=3,
        max_running_seqs=2,
        max_cache_len=8,
        max_prefill_tokens=8,
        kv_block_size=4,
        num_cache_blocks=4,
    )
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state,
            SamplingParams(),
            eos_token_id=None,
            prompt_indices=[0, 1, 2, 3],
        )

    assert state.generated == [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
    assert state.finish_reason == ["length", "length", "length", "length"]


def test_no_sync_rollout_drains_pending_prompts_when_first_token_hits_length_cap():
    """Rows with max_new_tokens=1 should finish during prefill and free slots."""

    manager = _FakeInferenceManager()
    state = InferenceBatchState(
        prompts=[[10], [11], [12], [13]],
        max_new_tokens=1,
        max_running_seqs=2,
        max_cache_len=4,
        max_prefill_tokens=8,
        kv_block_size=4,
        num_cache_blocks=2,
    )
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state,
            SamplingParams(),
            eos_token_id=None,
            prompt_indices=[0, 1, 2, 3],
        )

    assert state.generated == [[1], [1], [1], [1]]
    assert state.finish_reason == ["length", "length", "length", "length"]


def test_prefill_chunking_uses_global_max_running_prompts():
    """A 256-row flat rollout should stay one chunk even when dp_size is 8."""

    prompts = [[idx] for idx in range(256)]

    chunks = _chunk_prompts_for_prefill_budget(
        prompts,
        max_running_prompts=256,
        dp_size=8,
        max_prefill_tokens=1024,
    )

    assert [len(chunk) for chunk in chunks] == [256]


def test_decode_progress_log_is_worker_aggregated(monkeypatch):
    """Concurrent rollout progress should be throttled per worker, not per rollout."""

    manager = _FakeInferenceManager()
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)
    messages = []
    monkeypatch.setattr(inference_mod.logger, "info", lambda msg, *args: messages.append(msg % args))

    with PatchedContext(inference_mod, get_tp_context=lambda: ctx):
        manager._record_decode_progress(
            enabled=True,
            interval_s=0.0,
            rollout_key=1,
            active_count=4,
            token_delta=4,
            cache_tokens=8,
            sample_step=2,
            max_new_tokens=16,
        )
        manager._record_decode_progress(
            enabled=True,
            interval_s=60.0,
            rollout_key=2,
            active_count=6,
            token_delta=6,
            cache_tokens=9,
            sample_step=2,
            max_new_tokens=16,
        )

    assert len(messages) == 1
    assert "concurrent_rollouts=2" in messages[0]
    assert "active=10" in messages[0]
