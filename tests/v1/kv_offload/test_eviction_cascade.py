# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write-through eviction: a primary-tier victim reaches the slower tier first.

The contract these tests pin down:

1. A block evicted from the RAM tier is submitted to the secondary tier BEFORE
   its slot is released, and the slot is released only once the copy is
   confirmed (``pool full`` test).
2. A copy that fails or is refused is reported and counted as lost, with the
   slot released anyway — a broken disk must not pin the whole pool, and the
   block must never be reported as present (``copy fails`` test).
3. A promotion into a full primary tier does not consume a slot that a copy is
   still reading; it is reported as a miss and succeeds on a later step
   (``partial promotion`` test).
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
    """Secondary tier whose jobs finish only when the test says so."""

    def __init__(self, primary_kv_view, tier_type="fake"):
        super().__init__(
            offloading_spec=None,  # type: ignore[arg-type]
            primary_kv_view=primary_kv_view,
            tier_type=tier_type,
        )
        self.stored: set[OffloadKey] = set()
        self.pending: dict[int, tuple[JobMetadata, bool]] = {}
        self._next_id = 0
        self.store_calls: list[list[OffloadKey]] = []

    # --- SecondaryTierManager -------------------------------------------
    def lookup(self, key, req_context=None):
        return key in self.stored

    def submit_store(self, job_metadata: JobMetadata) -> None:
        self.store_calls.append(list(job_metadata.keys))
        self.pending[job_metadata.job_id] = (job_metadata, True)

    def submit_load(self, job_metadata: JobMetadata) -> None:
        self.pending[job_metadata.job_id] = (job_metadata, True)

    def get_finished_jobs(self):
        done = list(self.pending.items())
        self.pending.clear()
        results = []
        for job_id, (metadata, ok) in done:
            if ok:
                self.stored.update(metadata.keys)
            results.append(JobResult(job_id=job_id, success=ok))
        return results

    def on_new_request(self, req_context):
        return RequestOffloadingContext()

    # --- test helpers ----------------------------------------------------
    def finish(self, ok: bool = True) -> None:
        for job_id in list(self.pending):
            metadata, _ = self.pending[job_id]
            self.pending[job_id] = (metadata, ok)


def _mock_region(num_blocks: int):
    class _Region:
        def __init__(self, blocks: int):
            self._view = memoryview(
                torch.zeros((blocks, _ROW_BYTES), dtype=torch.int8).numpy()
            )

        def create_kv_memoryview(self):
            return self._view

    return _Region(num_blocks)


def _build(num_blocks: int):
    region = _mock_region(num_blocks)
    # Mirrors TieringOffloadingSpec: a grouped pool admits the leading part of
    # an oversized batch and defers the tail.
    primary = CPUPrimaryTierOffloadingManager(
        num_blocks=num_blocks, mmap_region=region, partial_admit=True
    )
    tier = _ControllableTier(primary.get_kv_memoryview())
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


def test_pool_full_writes_victim_before_freeing_the_slot():
    primary, tier, manager = _build(num_blocks=2)
    _fill(manager, primary, to_keys(range(2)))
    tier.pending.clear()  # cascade jobs settled by _fill; start clean
    assert tier.stored == set(to_keys(range(2)))

    # The pool is full: storing a third block must evict a victim, and the
    # victim must reach the secondary tier before the slot is reusable.
    new_keys = to_keys([2])
    output = manager.prepare_store(new_keys, _CTX)

    assert output is None  # nothing admitted this step
    assert tier.store_calls[-1] == to_keys([0])  # the LRU victim was submitted
    assert primary.held_blocks() == 1  # its slot is still allocated
    assert manager.primary_tier.lookup(to_keys([0])[0], _CTX) is False

    # Confirm the copy: the slot comes back and the retry is admitted.
    tier.finish(ok=True)
    _settle(manager)
    assert primary.held_blocks() == 0

    output = manager.prepare_store(new_keys, _CTX)
    assert output is not None
    assert list(output.keys_to_store) == new_keys
    assert not output.evicted_keys

    # The evicted block is still reachable, now from the slower tier.
    assert tier.lookup(to_keys([0])[0], _CTX) is True


def test_failed_copy_frees_the_slot_and_is_counted_as_lost():
    primary, tier, manager = _build(num_blocks=2)
    _fill(manager, primary, to_keys(range(2)))
    tier.pending.clear()

    # Model the realistic case: the victim's earlier cascade never landed (the
    # FS tier rejected it), so it is only in the RAM tier.
    tier.stored.clear()

    output = manager.prepare_store(to_keys([2]), _CTX)
    assert output is None
    assert primary.held_blocks() == 1

    # The write fails (disk full / I/O error). The block must not linger as
    # pinned, must not be claimed as stored, and must be counted as lost.
    tier.finish(ok=False)
    _settle(manager)

    assert primary.held_blocks() == 0
    assert tier.lookup(to_keys([0])[0], _CTX) is False
    assert manager._stats["evict_lost"] == 1
    assert manager._stats["evict_cascaded"] == 1  # the eviction-path cascade

    # The pool is usable again rather than pinned by the failure.
    output = manager.prepare_store(to_keys([2]), _CTX)
    assert output is not None
    assert list(output.keys_to_store) == to_keys([2])


def test_promotion_does_not_steal_a_slot_held_by_a_copy():
    primary, tier, manager = _build(num_blocks=2)
    _fill(manager, primary, to_keys(range(2)))

    # Candidate for promotion: present only in the slower tier.
    assert tier.lookup(to_keys([9])[0], _CTX) is False
    tier.stored.add(to_keys([9])[0])

    # Fill the pool and start a victim copy so no slot is free.
    tier.pending.clear()
    assert manager.prepare_store(to_keys([2]), _CTX) is None
    assert primary.held_blocks() == 1

    # The promotion cannot start: treating it as a hit would hand the GPU a
    # slot whose bytes are being read by the eviction copy.
    assert manager.lookup(to_keys([9])[0], _CTX) is False
    assert primary.held_blocks() == 1

    # After the copy lands there is room, and the promotion goes through.
    tier.finish(ok=True)
    _settle(manager)
    assert manager.lookup(to_keys([9])[0], _CTX) is None  # promotion started
    _settle(manager)
    assert manager.lookup(to_keys([9])[0], _CTX) is True


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
    assert manager.prepare_store(to_keys([2]), _CTX) is None
    assert len(tier.store_calls) == calls_before + 1
    # The victim was seen before by this tier: it is submitted again and the
    # tier answers with its own no-op, never with a silent skip here.
    assert tier.store_calls[-1] == to_keys([0])


def test_cascaded_keys_are_reported_to_the_caller():
    primary, tier, manager = _build(num_blocks=1)
    _fill(manager, primary, to_keys([0]))
    tier.pending.clear()

    output = manager.prepare_store(to_keys([1, 2]), _CTX)
    # With a single slot and its only victim in flight, nothing can be admitted
    # and the caller is told to retry — never "everything is stored".
    assert output is None
    assert primary.held_blocks() == 1

    tier.finish(ok=True)
    list(manager.take_events())
    output = manager.prepare_store(to_keys([1, 2]), _CTX)
    assert output is not None
    assert len(output.keys_to_store) == 1
    assert list(output.deferred_keys) == to_keys([2])
