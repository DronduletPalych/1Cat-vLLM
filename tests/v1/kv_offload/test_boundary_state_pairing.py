# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Mamba boundary-state store path.

A hybrid block is reusable only when its state and its attention KV are both
in the tier: the prefix-cache hit length is the minimum over groups, so a state
stored without its attention half can never be restored and only burns FS
state capacity. These tests pin that pairing down:

* a boundary state is held back (not written, not dropped) while its attention
  half is missing, and is stored together with it once it appears;
* the positional ``block_ids`` view keeps up with the allocation even while a
  transfer is in flight, so a materialised block never looks absent;
* the state cap prefers never-read checkpoints over the oldest one;
* one state group with nothing in the served window still cancels the chain.

They build the connector scheduler directly (no engine, no GPU) and use a
single-block-size layout, so they run next to the tier tests.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    OffloadingConnectorScheduler,
    RequestOffloadState,
    SchedulerOffloadConfig,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    OffloadingManager,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.tiering.fs.capacity import FS_BLOCK, DirCapacity

_BLOCK = 16  # tokens per GPU block == tokens per offloaded block
_ENTRY = FS_BLOCK * 2
_FILE_BYTES = 4 * 1024

_MAMBA_GROUPS = (0, 1)
_ATTN_GROUP = 2


def _group_config(group_idx: int, *, mamba: bool) -> GroupOffloadConfig:
    return GroupOffloadConfig(
        group_idx=group_idx,
        gpu_block_size=_BLOCK,
        offloaded_block_size=_BLOCK,
        hash_block_size_factor=_BLOCK // 4,
        sliding_window_size_in_blocks=1 if mamba else None,
        requires_exact_boundary_source=mamba,
    )


def _empty_output(**overrides) -> SimpleNamespace:
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set()
        ),
        num_scheduled_tokens={},
        finished_req_ids=set(),
        preempted_req_ids=set(),
        boundary_state_offloads=None,
    )
    for name, value in overrides.items():
        setattr(output, name, value)
    return output


def _make_scheduler(num_blocks: int = 4) -> OffloadingConnectorScheduler:
    """A two-Mamba-group plus one-attention-group scheduler for one request."""
    scheduler = object.__new__(OffloadingConnectorScheduler)
    scheduler.config = SchedulerOffloadConfig(
        kv_group_configs=(
            _group_config(0, mamba=True),
            _group_config(1, mamba=True),
            _group_config(_ATTN_GROUP, mamba=False),
        ),
        block_size_factor=1,
        num_workers=1,
        offload_prompt_only=False,
        num_kv_cache_groups=3,
    )
    scheduler._group_config_by_idx = {
        group.group_idx: group for group in scheduler.config.kv_group_configs
    }
    scheduler._sliding_window_groups = _MAMBA_GROUPS
    scheduler._lookup_groups = _MAMBA_GROUPS + (_ATTN_GROUP,)
    scheduler._mamba_align_size = None
    scheduler.manager = MagicMock(spec=OffloadingManager)
    scheduler.manager.prepare_store.side_effect = lambda keys, ctx: (
        generate_store_output(keys)
    )

    request = MagicMock()
    request.request_id = "req"
    request.kv_transfer_params = None
    request.block_hashes = [
        BlockHash(f"h{idx}".encode()) for idx in range(num_blocks * 4)
    ]
    request.num_computed_tokens = 0
    request.num_prompt_tokens = num_blocks * _BLOCK
    request.num_tokens = num_blocks * _BLOCK
    request.shared_prefix_boundary = 0
    request.is_finished.return_value = False
    req_status = RequestOffloadState(
        config=scheduler.config,
        req=request,
        req_context=ReqContext(req_id=request.request_id),
        offloading_context=RequestOffloadingContext(),
    )
    req_status.update_offload_keys()
    scheduler._req_status = {"req": req_status}
    scheduler._current_batch_load_jobs = {}
    scheduler._current_batch_jobs_to_flush = set()
    scheduler._blocks_being_loaded = set()
    scheduler._job_counter = 0
    scheduler._stale_job_threshold = 0
    scheduler._jobs = {}
    scheduler._block_id_to_pending_jobs = {}
    scheduler._defer_stats = {
        "deferred_stores": 0,
        "deferred_promotions": 0,
        "store_cannot_proceed": 0,
        "store_deferred": 0,
    }
    scheduler._defer_last_log = 0.0
    scheduler._boundary_pre = 0
    scheduler._boundary_skips = {"not_ready": 0, "busy": 0, "retrycap": 0}
    scheduler._max_pending_boundary_offloads = 8
    return scheduler


def _src_block_ids(job) -> list[int]:
    src_spec, _dst_spec = job.transfer_spec
    assert isinstance(src_spec, GPULoadStoreSpec)
    return list(src_spec.block_ids)


def test_boundary_state_waits_for_its_attention_half():
    """(a) No state-only write, and the offer is not lost."""
    scheduler = _make_scheduler()
    req_status = scheduler._req_status["req"]
    # Only the first attention block is materialised; the boundary needs the
    # second one (boundary 32 tokens -> block index 1).
    req_status.group_states[_ATTN_GROUP].block_ids[:] = [501]

    first = _empty_output(
        num_scheduled_tokens={"req": 32},
        boundary_state_offloads={"req": [(0, 91, 32)]},
    )
    meta = scheduler.build_connector_meta(first)

    assert meta.store_jobs == {}
    assert scheduler.manager.prepare_store.call_count == 0
    assert req_status.pending_boundary_offloads == [(0, 91, 32)]
    assert scheduler._boundary_skips["not_ready"] >= 1


def test_boundary_state_is_stored_with_its_attention_half_when_it_appears():
    """(a) The retried offer is written as one bundle, state + attention."""
    scheduler = _make_scheduler()
    req_status = scheduler._req_status["req"]
    req_status.group_states[0].block_ids[:] = [90, 91]
    req_status.group_states[1].block_ids[:] = [190, 191]
    req_status.group_states[_ATTN_GROUP].block_ids[:] = [501, 502]

    output = _empty_output(
        num_scheduled_tokens={"req": 32},
        boundary_state_offloads={"req": [(0, 91, 32)]},
    )
    meta = scheduler.build_connector_meta(output)

    assert len(meta.store_jobs) == 1
    [job] = meta.store_jobs.values()
    assert _src_block_ids(job) == [91, 502]
    src_spec, _dst_spec = job.transfer_spec
    assert src_spec.group_sizes == [1, 0, 1]
    assert src_spec.block_indices == [1, 0, 1]
    # Both halves were handed to the tier in one job.
    assert scheduler.manager.prepare_store.call_count == 1
    (offered_keys, _ctx) = scheduler.manager.prepare_store.call_args.args
    assert len(offered_keys) == 2
    assert req_status.pending_boundary_offloads == []


def test_block_ids_keep_up_while_a_transfer_is_in_flight():
    """(b) The positional list follows the allocation, not the store path."""
    scheduler = _make_scheduler()
    req_status = scheduler._req_status["req"]
    # A load owns the request's transfer slot for this step.
    scheduler._jobs[7] = SimpleNamespace(is_store=False)
    req_status.transfer_jobs.add(7)

    output = _empty_output(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["req"],
            new_block_ids=[([801], [802], [803])],
            resumed_req_ids=set(),
        ),
        num_scheduled_tokens={"req": _BLOCK},
    )
    meta = scheduler.build_connector_meta(output)

    assert meta.store_jobs == {}
    assert req_status.group_states[0].block_ids == [801]
    assert req_status.group_states[1].block_ids == [802]
    assert req_status.group_states[_ATTN_GROUP].block_ids == [803]


def test_state_cap_victim_prefers_never_read_checkpoint(tmp_path):
    """(c) A read checkpoint is kept; the oldest never-read one goes."""
    groups = list(_MAMBA_GROUPS) + [_ATTN_GROUP]
    cap = DirCapacity(
        str(tmp_path),
        max_bytes=64 * len(groups) * _ENTRY,
        max_overshoot_bytes=_ENTRY,
        group_indices=groups,
        state_max_bytes=2 * len(_MAMBA_GROUPS) * _ENTRY,
        mamba_groups=list(_MAMBA_GROUPS),
    )
    try:

        def checkpoint(n: int) -> dict[int, str]:
            paths = {}
            for group in groups:
                block_hash = f"{n:032x}"
                path = os.path.join(
                    str(tmp_path),
                    "cache_r0",
                    block_hash[:3],
                    f"{block_hash[:2]}_g{group}",
                    block_hash + ".bin",
                )
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as fh:
                    fh.write(b"x" * _FILE_BYTES)
                assert cap.try_reserve(path, _ENTRY)
                cap.commit_store(path, _ENTRY)
                paths[group] = path
            return paths

        first = checkpoint(0)
        second = checkpoint(1)
        # A chain hit reads every group file of the checkpoint it restores.
        for group in groups:
            cap.touch(first[group])

        third = checkpoint(2)

        assert all(os.path.exists(path) for path in first.values())
        assert not os.path.exists(second[_MAMBA_GROUPS[0]])
        assert all(os.path.exists(path) for path in third.values())
        snap = cap.snapshot()
        assert snap["fs_state_evictions_total"] == 1
        assert snap["fs_state_evict_unread_total"] == 1
        assert snap["fs_state_evict_read_total"] == 0
    finally:
        cap.shutdown()


def _lookup_scheduler(present: dict[int, set[int]], num_blocks: int = 4):
    """Scheduler whose tier holds *present[group]* offload blocks of a request."""
    scheduler = _make_scheduler(num_blocks=num_blocks)
    req_status = scheduler._req_status["req"]
    keys_by_group = {
        group: req_status.group_states[group].offload_keys for group in range(3)
    }
    held = {
        keys_by_group[group][idx]
        for group, indices in present.items()
        for idx in indices
    }
    scheduler.manager.lookup.side_effect = (
        lambda key, ctx: key in held
    )
    scheduler.manager.contains.side_effect = lambda key: key in held
    return scheduler, req_status


def test_one_state_group_without_a_window_cancels_the_chain():
    """(d) Invariant: nothing in the state window means no chain hit at all."""
    everything = {0: set(range(4)), 1: set(range(4)), _ATTN_GROUP: set(range(4))}
    scheduler, req_status = _lookup_scheduler(everything)
    # The last prompt token is always recomputed, so the ceiling is one short
    # of the prompt: the chain still covers all four blocks.
    assert scheduler._lookup(req_status) == 4 * _BLOCK - 1

    # The attention chain is complete, but the state group holds no block in
    # the served window: the request must not be told it has a hit.
    missing_state = {0: set(range(4)), 1: set(), _ATTN_GROUP: set(range(4))}
    scheduler, req_status = _lookup_scheduler(missing_state)
    assert scheduler._lookup(req_status) == 0
