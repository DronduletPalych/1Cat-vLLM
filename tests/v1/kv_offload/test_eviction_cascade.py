# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write-through eviction: a primary-tier victim reaches the slower tier first.

The contract these tests pin down:

1. A block evicted from the RAM tier is submitted to the secondary tier BEFORE
   its slot is reused, and the slot is reused only once the copy is confirmed
   (``full pool`` test). The store that needed the slot is accepted in the same
   call: it waits, bounded, for the copy its own eviction started instead of
   being refused and retried by a caller that may have no step left.
2. A copy that fails is reported and counted as lost, with the slot released
   anyway — a broken disk must not pin the whole pool, and the block must never
   be reported as present (``copy fails`` test).
3. A copy that misses the budget, or a pool whose blocks are all pinned by
   transfers, makes the store refuse; that refusal is counted as lost instead of
   hiding behind ``lost=0`` (``pinned pool`` test).
4. A promotion into a full primary tier evicts and waits like any other store,
   so a filesystem hit is not thrown away just because the pool is full
   (``promotion`` test).
"""

import torch

from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

_CTX = ReqContext(req_id="evict-cascade")
_ROW_BYTES = 16


def to_keys(numbers) -> list[OffloadKey]:
    return [make_offload_key(str(i).encode(), 0) for i in numbers]


class _ControllableTier(SecondaryTierManager):
    """Secondary tier whose copies land immediately, or only when told."""

    def __init__(self, primary_kv_view, tier_type="fake", auto=True):
        super().__init__(
            offloading_spec=None,  # type: ignore[arg-type]
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
        )
        self.auto = auto
        self.fail_next_store = False
        self.fail_next_load = False
        self.stored: set[OffloadKey] = set()
        self.pending: dict[int, tuple[JobMetadata, bool]] = {}
        self.finished: list[tuple[int, bool]] = []
        self.store_calls: list[list[OffloadKey]] = []
        self.load_calls: list[list[OffloadKey]] = []

    # --- SecondaryTierManager -------------------------------------------
    def lookup(self, key, req_context=None):
        return key in self.stored

    def submit_store(self, job_metadata: JobMetadata) -> None:
        self.store_calls.append(list(job_metadata.keys))
        self._queue(job_metadata, not self.fail_next_store)

    def submit_load(self, job_metadata: JobMetadata) -> None:
        self.load_calls.append(list(job_metadata.keys))
        self._queue(job_metadata, not self.fail_next_load)

    def get_finished_jobs(self):
        results = [
            JobResult(job_id=job_id, success=success)
            for job_id, success in self.finished
        ]
        self.finished.clear()
        return results

    def on_new_request(self, req_context):
        return RequestOffloadingContext()

    # --- test helpers ----------------------------------------------------
    def _queue(self, job_metadata: JobMetadata, ok: bool) -> None:
        self.pending[job_metadata.job_id] = (job_metadata, ok)
        if self.auto:
            self.finish(ok)

    def finish(self, ok: bool | None = None) -> None:
        """Complete every queued copy now (optionally forcing its outcome)."""
        for job_id, (metadata, success) in list(self.pending.items()):
            self.pending.pop(job_id)
            success = success if ok is None else ok
            if success:
                self.stored.update(metadata.keys)
            self.finished.append((job_id, success))


def _mock_region(num_blocks: int):
    class _Region:
        def __init__(self, blocks: int):
            self._view = memoryview(
                torch.zeros((blocks, _ROW_BYTES), dtype=torch.int8).numpy()
            )

        def create_kv_memoryview(self):
            return self._view

    return _Region(num_blocks)


def _build(num_blocks: int, auto: bool = True):
    region = _mock_region(num_blocks)
    # Mirrors TieringOffloadingSpec: a grouped pool admits the leading part of
    # an oversized batch and defers the tail.
    primary = CPUPrimaryTierOffloadingManager(
        num_blocks=num_blocks, mmap_region=region, partial_admit=True
    )
    tier = _ControllableTier(primary.get_kv_memoryview(), auto=auto)
    manager = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[tier])
    return primary, tier, manager


def _settle(manager, attempts=8):
    """Run the per-step poll until no transfer job of this manager is left.

    _maybe_process_finished_jobs() runs at most once per engine step, so a job
    submitted inside a step is only observed by the following step's poll.
    """
    for _ in range(attempts):
        list(manager.take_events())
        if not manager._transfer_jobs:
            return
    raise AssertionError("transfer jobs did not settle")


def _fill(manager, primary, keys):
    """Store keys into an empty pool and settle every cascade job."""
    output = manager.prepare_store(keys, _CTX)
    assert output is not None
    assert list(output.keys_to_store) == list(keys)
    manager.complete_store(list(keys), _CTX, success=True)
    _settle(manager)
    for key in keys:
        block = primary._policy.get(key)
        assert block is not None and block.ref_cnt == 0
    return output


def test_pool_full_writes_victim_before_reusing_its_slot():
    primary, tier, manager = _build(num_blocks=2)
    _fill(manager, primary, to_keys(range(2)))
    tier.pending.clear()  # cascade jobs settled by _fill; start clean
    assert tier.stored == set(to_keys(range(2)))

    # The pool is full: storing a third block must evict a victim, persist it,
    # and reuse its slot inside the same call.
    new_keys = to_keys([2])
    output = manager.prepare_store(new_keys, _CTX)

    assert output is not None
    assert list(output.keys_to_store) == new_keys
    assert tier.store_calls[-1] == to_keys([0])  # the LRU victim was submitted
    assert tier.lookup(to_keys([0])[0], _CTX) is True  # and it landed
    assert manager.primary_tier.lookup(to_keys([0])[0], _CTX) is False
    # The store waited for the copy instead of being refused, and nothing was
    # dropped on the way.
    assert primary.held_blocks() == 0
    assert primary.refusal_stats() == {"batches": 0, "blocks": 0, "lost": 0}
    assert manager._stats["evict_cascaded"] == 1
    assert manager._stats["evict_lost"] == 0
    assert manager._stats["evict_wait_calls"] >= 1

    # The slot is really reusable: the next block is admitted (evicting a new
    # victim, since the pool is full again) without any refusal.
    tier.store_calls.clear()
    output = manager.prepare_store(to_keys([3]), _CTX)
    assert output is not None
    assert list(output.keys_to_store) == to_keys([3])
    assert tier.store_calls == [to_keys([1])]
    assert primary.refusal_stats() == {"batches": 0, "blocks": 0, "lost": 0}


def test_copy_that_misses_the_budget_refuses_and_counts_the_loss():
    primary, tier, manager = _build(num_blocks=2)
    _fill(manager, primary, to_keys(range(2)))
    # From here on the tier holds its copies: they cannot land inside the pool's
    # budget, and the store must not silently disappear.
    primary.reclaim_timeout = 0.02
    tier.auto = False

    output = manager.prepare_store(to_keys([2]), _CTX)
    assert output is None
    assert tier.store_calls[-1] == to_keys([0])  # the victim was still submitted
    assert primary.held_blocks() == 1  # and its slot stays pinned by the copy

    # The refusal is visible: one batch, one block, counted as lost.
    assert primary.refusal_stats() == {"batches": 1, "blocks": 1, "lost": 1}
    assert manager._stats["evict_lost"] == 0  # the copy itself did not fail

    # A second offer does not evict a second victim for the same admission.
    assert manager.prepare_store(to_keys([2]), _CTX) is None
    assert len(tier.store_calls) == 2  # 1 batch from _fill + 1: no churn

    # Once the copy lands the slot comes back and the block is admitted.
    tier.auto = True
    tier.finish(ok=True)
    _settle(manager)
    assert primary.held_blocks() == 0
    output = manager.prepare_store(to_keys([2]), _CTX)
    assert output is not None
    assert list(output.keys_to_store) == to_keys([2])


def test_failed_copy_frees_the_slot_and_is_counted_as_lost():
    primary, tier, manager = _build(num_blocks=2)
    _fill(manager, primary, to_keys(range(2)))
    tier.pending.clear()

    # Model the realistic case: the victim's earlier cascade never landed (the
    # FS tier rejected it), so it is only in the RAM tier.
    tier.stored.clear()
    tier.fail_next_store = True

    # The write fails (disk full / I/O error). The block must not linger as
    # pinned, must not be claimed as stored, must be counted as lost, and the
    # waiting store must still get the slot the failure released.
    output = manager.prepare_store(to_keys([2]), _CTX)

    assert output is not None
    assert list(output.keys_to_store) == to_keys([2])
    assert primary.held_blocks() == 0
    assert tier.lookup(to_keys([0])[0], _CTX) is False
    assert manager._stats["evict_lost"] == 1
    assert manager._stats["evict_cascaded"] == 1  # the eviction-path cascade
    assert primary.refusal_stats()["blocks"] == 0


def test_all_pinned_pool_refuses_and_counts_the_loss():
    """No evictable block: the refusal is the only correct answer, and it is

    counted instead of vanishing into lost=0.
    """
    primary, tier, manager = _build(num_blocks=2)
    keys = to_keys(range(2))
    _fill(manager, primary, keys)
    tier.pending.clear()
    tier.store_calls.clear()

    # Both slots are used as transfer sources (an in-flight GPU load): the
    # policy refuses to evict them.
    spec = primary.prepare_load(keys, _CTX)
    assert spec is not None

    output = manager.prepare_store(to_keys([2]), _CTX)
    assert output is None
    assert tier.store_calls == []  # nothing could be evicted, nothing written
    assert primary.refusal_stats() == {"batches": 1, "blocks": 1, "lost": 1}
    assert primary.free_slots() == 0

    # Pinning is what blocked it: release the loads and the store goes through.
    primary.complete_load(keys, _CTX)
    output = manager.prepare_store(to_keys([2]), _CTX)
    assert output is not None
    assert list(output.keys_to_store) == to_keys([2])


def test_promotion_evicts_and_waits_instead_of_missing():
    primary, tier, manager = _build(num_blocks=2)
    _fill(manager, primary, to_keys(range(2)))
    tier.pending.clear()
    tier.store_calls.clear()

    # Candidate for promotion: present only in the slower tier.
    promoted = to_keys([9])
    assert tier.lookup(promoted[0], _CTX) is False
    tier.stored.add(promoted[0])

    # The pool is full, so the promotion has to evict and reuse a victim's slot.
    assert manager.lookup(promoted[0], _CTX) is None  # promotion started
    assert tier.store_calls[-1] == to_keys([0])  # the LRU victim was persisted
    assert tier.lookup(to_keys([0])[0], _CTX) is True

    _settle(manager)  # flushes the batched submit_load and completes it
    assert tier.load_calls[-1] == promoted
    assert manager.lookup(promoted[0], _CTX) is True
    # The promotion was not refused for lack of a slot.
    assert manager._stats["promo_refused_blocks"] == 0


def test_deduplication_leaves_tier_side_dedupe_in_charge():
    """A victim already on disk is re-submitted; the tier skips the write.

    ``submit_store`` owns the on-disk check (a file that exists is not
    rewritten), because only the tier knows whether its own eviction dropped the
    file. The manager must not keep a second, staler mirror of that state.
    """
    primary, tier, manager = _build(num_blocks=2)
    _fill(manager, primary, to_keys(range(2)))
    tier.pending.clear()

    calls_before = len(tier.store_calls)
    output = manager.prepare_store(to_keys([2]), _CTX)
    assert output is not None
    assert len(tier.store_calls) == calls_before + 1
    # The victim was seen before by this tier: it is submitted again and the
    # tier answers with its own no-op, never with a silent skip here.
    assert tier.store_calls[-1] == to_keys([0])


def test_cascaded_keys_are_reported_to_the_caller():
    primary, tier, manager = _build(num_blocks=1)
    _fill(manager, primary, to_keys([0]))
    tier.pending.clear()

    # One slot, one victim: the victim's slot is reused for the first key and
    # the rest of the batch is deferred (never "everything is stored").
    output = manager.prepare_store(to_keys([1, 2]), _CTX)
    assert output is not None
    assert list(output.keys_to_store) == to_keys([1])
    assert list(output.deferred_keys) == to_keys([2])
    assert output.cascaded_keys == tuple(to_keys([0]))
