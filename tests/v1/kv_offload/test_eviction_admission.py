# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission into a full grouped pool: evict and wait, never refuse and drop.

Pinned here:

1. ``full pool accepts the block`` — with a full pool the store is NOT refused:
   the LRU victim is cascaded to the filesystem tier, its slot is reused by the
   caller's block, and no block is reported lost.
2. ``all pinned counts as loss`` — when nothing can be evicted the refusal is
   the only option, and it must appear in the per-group ``refused``/``lost``
   counters instead of leaving the channels invisible behind ``lost=0``.
3. The per-group counters keep ``out`` (confirmed stores) apart from ``casc``
   (confirmed eviction copies), so ``out`` can never exceed ``adm`` again.
"""

import torch

from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.manager import GroupedCPUOffloadingManager
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)

_CTX = ReqContext(req_id="admission")
_ROW_BYTES = 16
# Columns of GroupedCPUOffloadingManager._stats.
IN, NEW, ADM, DEF, OUT, LOST, REFUSED, CASC, REF = range(9)


def key(group: int, number: int) -> OffloadKey:
    return make_offload_key(number.to_bytes(8, "big"), group)


class _Tier(SecondaryTierManager):
    """Secondary tier that completes immediately or only when told."""

    def __init__(self, primary_kv_view, auto=True):
        super().__init__(
            offloading_spec=None,  # type: ignore[arg-type]
            primary_kv_view=primary_kv_view,
            tier_type="fake",
        )
        self.auto = auto
        self.stored: set[OffloadKey] = set()
        self.pending: dict[int, tuple[JobMetadata, bool]] = {}
        self.finished: list[tuple[int, bool]] = []
        self.store_calls: list[list[OffloadKey]] = []

    def lookup(self, key, req_context=None):
        return key in self.stored

    def submit_store(self, job_metadata: JobMetadata) -> None:
        self.store_calls.append(list(job_metadata.keys))
        self.pending[job_metadata.job_id] = (job_metadata, True)
        if self.auto:
            self.finish()

    def submit_load(self, job_metadata: JobMetadata) -> None:
        self.pending[job_metadata.job_id] = (job_metadata, True)
        if self.auto:
            self.finish()

    def get_finished_jobs(self):
        results = [
            JobResult(job_id=job_id, success=success)
            for job_id, success in self.finished
        ]
        self.finished.clear()
        return results

    def on_new_request(self, req_context):
        return RequestOffloadingContext()

    def finish(self, ok: bool = True) -> None:
        for job_id, (metadata, _) in list(self.pending.items()):
            self.pending.pop(job_id)
            if ok:
                self.stored.update(metadata.keys)
            self.finished.append((job_id, ok))


class _Region:
    def __init__(self, blocks: int):
        self._view = memoryview(
            torch.zeros((blocks, _ROW_BYTES), dtype=torch.int8).numpy()
        )

    def create_kv_memoryview(self):
        return self._view


class _TierStub(CPUPrimaryTierOffloadingManager):
    """Keeps the region object alive for the lifetime of the test."""

    def shutdown(self) -> None:
        return None


def build(num_blocks: int, groups=(0, 1), auto=True):
    managers: dict[int, TieringOffloadingManager] = {}
    tiers: dict[int, _Tier] = {}
    for group in groups:
        primary = _TierStub(num_blocks=num_blocks, mmap_region=_Region(num_blocks))
        tier = _Tier(primary.get_kv_memoryview(), auto=auto)
        tiers[group] = tier
        managers[group] = TieringOffloadingManager(
            primary_tier=primary, secondary_tiers=[tier]
        )
    grouped = GroupedCPUOffloadingManager(managers)
    return grouped, managers, tiers


def fill(grouped, managers, tiers, group: int, numbers):
    keys = [key(group, n) for n in numbers]
    output = grouped.prepare_store(keys, _CTX)
    assert output is not None
    assert set(output.keys_to_store) == set(keys)
    grouped.complete_store(keys, _CTX, success=True)
    settle(managers[group])
    for k in keys:
        block = managers[group].primary_tier._policy.get(k)
        assert block is not None and block.ref_cnt == 0
    tiers[group].pending.clear()
    return keys


def settle(manager, attempts=8):
    for _ in range(attempts):
        list(manager.take_events())
        if not manager._transfer_jobs:
            return
    raise AssertionError("transfer jobs did not settle")


def test_full_pool_evicts_and_accepts_the_block_without_loss(monkeypatch):
    """Pool full: the block is admitted, the victim reaches the FS tier, loss=0."""
    monkeypatch.setenv("VLLM_TIER_STATS", "1")
    grouped, managers, tiers = build(num_blocks=2)
    for group in (0, 1):
        fill(grouped, managers, tiers, group, (0, 1))
    for tier in tiers.values():
        tier.store_calls.clear()

    new = [key(0, 2), key(1, 2)]
    output = grouped.prepare_store(new, _CTX)

    assert output is not None
    assert set(output.keys_to_store) == set(new)
    for group in (0, 1):
        # The LRU victim of each pool was written to the filesystem tier and
        # only then replaced by the new block.
        assert tiers[group].store_calls == [[key(group, 0)]]
        assert tiers[group].lookup(key(group, 0), _CTX) is True
        assert managers[group].primary_tier.lookup(key(group, 0), _CTX) is False
        # The new block owns the victim's slot (not ready yet: its GPU→CPU copy
        # has not been completed in this test).
        assert managers[group].primary_tier.contains(key(group, 2))
        stats = grouped._stats[group]
        assert stats[IN] == 3  # two filled blocks + one offered block
        assert stats[NEW] == 3  # every offer needed a slot (no re-offers)
        assert stats[ADM] == 3
        assert stats[REFUSED] == 0
        assert stats[REF] == 0
        assert stats[LOST] == 0
        assert stats[CASC] == 1
        assert managers[group].primary_tier.refusal_stats() == {
            "batches": 0,
            "blocks": 0,
            "lost": 0,
        }


def test_all_pinned_refusal_is_counted_in_the_group_counters(monkeypatch):
    """Nothing evictable: the refusal must show up as refused and lost."""
    monkeypatch.setenv("VLLM_TIER_STATS", "1")
    grouped, managers, tiers = build(num_blocks=2)
    for group in (0, 1):
        fill(grouped, managers, tiers, group, (0, 1))

    # Both slots of each pool are transfer sources, so the policy may not evict.
    for group in (0, 1):
        managers[group].primary_tier.prepare_load(
            [key(group, 0), key(group, 1)], _CTX
        )
    for tier in tiers.values():
        tier.store_calls.clear()

    output = grouped.prepare_store([key(0, 2), key(1, 2)], _CTX)

    assert output is None  # nothing admitted anywhere
    for group in (0, 1):
        stats = grouped._stats[group]
        assert stats[REF] == 1
        assert stats[REFUSED] == 1
        assert stats[LOST] == 1
        assert stats[CASC] == 0
        assert tiers[group].store_calls == []
        assert managers[group].primary_tier.refusal_stats() == {
            "batches": 1,
            "blocks": 1,
            "lost": 1,
        }


def test_out_counts_only_admitted_stores_cascades_are_separate(monkeypatch):
    """`out` never exceeds `adm`: eviction copies are booked under `casc`."""
    monkeypatch.setenv("VLLM_TIER_STATS", "1")
    grouped, managers, tiers = build(num_blocks=2, groups=(0,))
    fill(grouped, managers, tiers, 0, (0, 1))

    # Store a third block: one admitted store plus one cascaded victim.
    output = grouped.prepare_store([key(0, 2)], _CTX)
    assert output is not None
    grouped.complete_store([key(0, 2)], _CTX, success=True)
    settle(managers[0])

    stats = grouped._stats[0]
    assert stats[ADM] == 3
    assert stats[OUT] == 3  # three admitted stores confirmed, cascades apart
    assert stats[CASC] == 1
    assert stats[OUT] <= stats[ADM]
