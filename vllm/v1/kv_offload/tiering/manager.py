# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
TieringOffloadingManager: Multi-tier KV cache offloading orchestrator.

This manager coordinates between a CPU primary tier (with direct GPU access)
and zero or more secondary tiers (Storage, Network, etc.) to provide
hierarchical KV cache offloading.

Key Design Principles:
1. Always offload to all tiers — When a block is stored to the primary tier,
   it is cascaded to ALL secondary tiers
2. Primary tier is the gateway — Secondary tiers cannot access GPU memory
   directly; all data flows through the CPU primary tier
3. Staged promotion — Blocks in secondary tiers must be promoted to the
   primary tier before GPU can access them
4. Transparent retry mechanism — Return None from lookup() to signal
   "data is being promoted, try later"
5. ref_cnt as eviction protection — primary.prepare_read() increments ref_cnt,
   protecting blocks from eviction until complete_read() is called
"""

import os
import time
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    OffloadPolicy,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.tiering.base import (
    JobId,
    JobMetadata,
    SecondaryTierManager,
)

logger = init_logger(__name__)

# Optional tier-level hit/miss counters (env-gated diagnostics).
_TIER_STATS = os.environ.get("VLLM_TIER_STATS") == "1"


@dataclass
class PendingPromotion:
    """Accumulator for blocks awaiting submit_load() for one (tier, request)."""

    req_context: ReqContext
    keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)


@dataclass
class EvictionBatch:
    """Blocks evicted from the primary tier and being copied to the secondaries.

    Their primary slots stay allocated until every tier reports the copy done
    (or failed): the copy reads the slot, so freeing it earlier would write a
    later block over the bytes mid-flight. This is the write-through half of
    eviction — without it a victim that was never cascaded is lost for good.
    """

    keys: list[OffloadKey]
    block_ids: list[int]
    remaining: int
    success: bool = True


class CPUPrimaryTierOffloadingManager(CPUOffloadingManager):
    """CPUOffloadingManager with a primary/secondary transfer interface.

    The inherited prepare_store/complete_store/prepare_load/complete_load are the
    GPU-facing OffloadingManager interface. These aliases expose the same operations
    from the secondary tier perspective, where read/write refers to secondary
    accessing primary. This avoids confusion when reading TieringOffloadingManager
    code (e.g. calling prepare_load inside a cascade/store path would be misleading).
    """

    def __init__(
        self,
        num_blocks: int,
        mmap_region: SharedOffloadRegion,
        cache_policy: str = "lru",
        enable_events: bool = False,
        partial_admit: bool = False,
        group_label: str = "-",
    ):
        super().__init__(
            num_blocks=num_blocks,
            cache_policy=cache_policy,  # type: ignore[arg-type]
            enable_events=enable_events,
            # Secondary tiers promote asynchronously; the scheduler must retry
            # rather than treat an in-flight promotion as a miss.
            defer_inflight_lookup=True,
            partial_admit=partial_admit,
            group_label=group_label,
        )
        self._mmap_region = mmap_region
        # read/write is for CPU<->secondary transfers,
        # load/store is for CPU<->GPU transfers.
        # These aliases avoid calling prepare_load inside a store path.
        self.prepare_read = self.prepare_load
        self.complete_read = self.complete_load
        self.prepare_write = self.prepare_store
        self.complete_write = self.complete_store

        self._kv_memoryview = mmap_region.create_kv_memoryview()

    def get_kv_memoryview(self) -> memoryview:
        """Return the memoryview over the primary tier's KV cache buffer.

        The view has shape (num_blocks, row_stride_bytes) and is backed by the
        SharedOffloadRegion mmap.  Secondary tiers address block *b* as
        ``view[b]``.
        """
        return self._kv_memoryview

    def shutdown(self) -> None:
        super().shutdown()
        self._kv_memoryview.release()
        self._mmap_region.cleanup()


class TieringOffloadingManager(OffloadingManager):
    """
    Orchestrates multi-tier KV cache offloading.

    This manager coordinates between a CPU primary tier (with direct GPU access)
    and zero or more secondary tiers (Storage, Network, etc.) to provide
    hierarchical KV cache offloading.

    Key internal state:
      - Minimal state tracking; relies on secondary tiers to report completion
        via get_finished_jobs()
      - Secondary tiers return JobResult objects containing all necessary
        information
      - job_id_counter: monotonically increasing counter for job IDs
    """

    def __init__(
        self,
        primary_tier: CPUPrimaryTierOffloadingManager,
        secondary_tiers: list[SecondaryTierManager] | None = None,
        enable_events: bool = False,
    ):
        """
        Initialize the TieringOffloadingManager.

        Args:
            primary_tier: The primary tier manager (CPU-based).
            secondary_tiers: List of secondary tier managers (e.g., Storage,
                            Network). Can be None or empty list.
            enable_events: Whether to track offloading events
        """
        self.primary_tier: CPUPrimaryTierOffloadingManager = primary_tier
        self.secondary_tiers = secondary_tiers or []

        self._job_id_counter: int = 0
        self.events: list[OffloadingEvent] | None = [] if enable_events else None

        # Job tracking: maps job_id to metadata for all in-flight transfers.
        # JobMetadata.is_promotion distinguishes direction:
        #   True:  secondary → primary (promotion)
        #   False: primary → secondary (cascade)
        self._transfer_jobs: dict[JobId, JobMetadata] = {}

        # Pending promotion requests accumulated during lookup() calls; flushed
        # as one batched submit_load() per (tier, request) in take_events().
        # Outer key: tier. Inner key: req_context.req_id — the same ReqContext
        # object is reused for all block lookups of a given request per engine step.
        self._pending_load_submissions: dict[
            SecondaryTierManager, dict[str, PendingPromotion]
        ] = {}

        # Gate for once-per-step execution of _maybe_process_finished_jobs().
        # Reset at the end of each step in take_events().
        self._processed_jobs_this_step: bool = False

        # Per-request set of secondary tiers that requested REQUEST_LEVEL
        # policy. Populated in on_new_request(),
        # cleaned up in on_request_finished().
        self._request_level_tiers: defaultdict[str, set[SecondaryTierManager]] = (
            defaultdict(set)
        )

        # Diagnostics: where lookups are answered from, and how many cascades
        # to the secondary tiers failed.
        self._stats: dict[str, int] = {
            "primary_hit": 0,
            "primary_inflight": 0,
            "sec_lookup": 0,
            "sec_hit": 0,
            "promo_ok": 0,
            "promo_fail": 0,
            "miss": 0,
            "store_failures": 0,
            "evict_cascaded": 0,
            "evict_lost": 0,
        }
        self._stats_last = time.monotonic()

        # Evicted blocks whose copy to the secondary tiers is in flight. Their
        # primary slots stay allocated until the copy lands.
        self._eviction_jobs: dict[JobId, EvictionBatch] = {}

        # Optional per-group sinks installed by the grouped pool, so the
        # in/out/lost accounting reaches the single place that logs it.
        self._store_outcome_sink: Callable[[int, int], None] | None = None
        self._hit_sink: Callable[[str], None] | None = None

        # Write-through on eviction: a victim is copied to the secondary tiers
        # BEFORE its primary slot is released. Without a secondary tier there is
        # nothing to copy to and the primary pool keeps its plain LRU behavior.
        if self.secondary_tiers:
            self.primary_tier.set_eviction_cascade(self._cascade_evicted)

    def set_store_outcome_sink(self, sink: Callable[[int, int], None]) -> None:
        """Install a per-group callback for (confirmed_in_fs, lost) block counts."""
        self._store_outcome_sink = sink

    def set_hit_sink(self, sink: Callable[[str], None]) -> None:
        """Install a per-group callback that reports which tier served a hit."""
        self._hit_sink = sink

    def _report_store_outcome(self, confirmed: int, lost: int) -> None:
        if self._store_outcome_sink is not None:
            self._store_outcome_sink(confirmed, lost)

    def _cascade_evicted(
        self,
        evicted: Sequence[tuple[OffloadKey, Any]],
        req_context: ReqContext,
    ) -> set[OffloadKey]:
        """Copy primary-tier victims to every secondary tier before freeing them.

        Called from the primary pool's eviction path. The victims are already
        unlinked from the primary policy; their slots are still allocated and
        hold the bytes, which is all a secondary tier needs to read them. The
        returned keys are the ones whose slot the pool must keep: it is released
        in _process_finished_jobs() once the copy is confirmed.

        The tiers deduplicate themselves (a block already on disk is not
        rewritten), so re-submitting a victim that a previous cascade persisted
        costs one stat() and no I/O.
        """
        if not self.secondary_tiers or not evicted:
            return set()

        keys = [key for key, _ in evicted]
        block_ids = np.array([int(block.block_id) for _, block in evicted], dtype=np.int64)
        batch = EvictionBatch(
            keys=keys,
            block_ids=[int(b) for b in block_ids],
            remaining=len(self.secondary_tiers),
        )
        submitted = 0
        for tier in self.secondary_tiers:
            job_id = self._next_job_id()
            job_metadata = JobMetadata(
                job_id=job_id,
                keys=keys,
                block_ids=block_ids,
                is_promotion=False,
                req_context=req_context,
            )
            self._transfer_jobs[job_id] = job_metadata
            self._eviction_jobs[job_id] = batch
            try:
                tier.submit_store(job_metadata)
            except Exception:
                logger.exception(
                    "OFFLOAD-EVICT-CASCADE-ERROR tier=%s job=%s keys=%d",
                    tier.tier_type,
                    job_id,
                    len(keys),
                )
                self._transfer_jobs.pop(job_id, None)
                self._eviction_jobs.pop(job_id, None)
                batch.remaining -= 1
                batch.success = False
            else:
                submitted += 1

        if batch.remaining == 0:
            # Every submit failed synchronously: nothing holds the slots.
            self._stats["evict_lost"] += len(keys)
            self._report_store_outcome(0, len(keys))
            return set()

        if submitted:
            self._stats["evict_cascaded"] += len(keys)
        return set(keys)

    def _next_job_id(self) -> JobId:
        """Generate a unique job ID for async transfer tracking."""
        job_id = self._job_id_counter
        self._job_id_counter += 1
        return job_id

    def _maybe_process_finished_jobs(self):
        """
        Poll secondary tiers for completed jobs (at most once per step).

        Guarded by _processed_jobs_this_step: the first call in an engine step
        does the actual polling; subsequent calls are no-ops. The flag is reset
        in take_events() at the end of each step.
        """
        if self._processed_jobs_this_step:
            return
        self._processed_jobs_this_step = True
        self._process_finished_jobs()

    def _process_finished_jobs(self):
        """
        Unconditionally poll all secondary tiers for completed jobs.

        This method:
        1. Calls get_finished_jobs() on each secondary tier
        2. For completed stores (primary→secondary): calls primary.complete_read()
           to decrement ref_cnt
        3. For completed loads (secondary→primary): calls primary.complete_write()
           to make blocks available
        """
        for i, tier in enumerate(self.secondary_tiers):
            for completed_job in tier.get_finished_jobs():
                job_id = completed_job.job_id
                job_metadata = self._transfer_jobs.pop(job_id, None)
                assert job_metadata is not None, (
                    f"Finished job_id {job_id} from tier #{i}"
                    f" ({tier.tier_type}) not in _transfer_jobs"
                )

                eviction_batch = self._eviction_jobs.pop(job_id, None)
                if eviction_batch is not None:
                    self._finish_eviction_job(tier, completed_job, eviction_batch)
                    continue

                if job_metadata.is_promotion:
                    # secondary→primary transfer (promotion) completed.
                    # Make blocks available in primary tier.
                    self.primary_tier.complete_write(
                        job_metadata.keys,
                        job_metadata.req_context,
                        completed_job.success,
                    )
                else:
                    # primary→secondary transfer completed.
                    if not completed_job.success:
                        # The cascade failed: nothing may be treated as stored
                        # in the secondary tier. The file is absent (the store
                        # path is atomic: tmp + rename) and the FS tier has
                        # already released its capacity reservation, so
                        # lookup() keeps reporting a miss for these keys.
                        self._stats["store_failures"] += 1
                        logger.warning(
                            "OFFLOAD-STORE-FAIL tier=%s job=%s keys=%d "
                            "store_failures=%d",
                            tier.tier_type,
                            job_id,
                            len(job_metadata.keys),
                            self._stats["store_failures"],
                        )
                    else:
                        # The blocks are on disk: this is the `out` number of
                        # the per-group save accounting.
                        self._report_store_outcome(len(job_metadata.keys), 0)
                    # Decrement ref_cnt on primary blocks (also on failure:
                    # the slots must become evictable again).
                    self.primary_tier.complete_read(
                        job_metadata.keys, job_metadata.req_context
                    )

    def _finish_eviction_job(
        self, tier: SecondaryTierManager, completed_job, batch: EvictionBatch
    ) -> None:
        """Conclude one eviction-copy job; free the slots when the batch is done.

        A failed copy is not silent: it is logged and counted as `lost`, and the
        slot is released anyway so a broken disk cannot pin the whole pool.
        """
        batch.remaining -= 1
        if not completed_job.success:
            batch.success = False
            self._stats["store_failures"] += 1
            logger.warning(
                "OFFLOAD-EVICT-STORE-FAIL tier=%s job=%s keys=%d store_failures=%d",
                tier.tier_type,
                completed_job.job_id,
                len(batch.keys),
                self._stats["store_failures"],
            )
        if batch.remaining > 0:
            return

        released = self.primary_tier.release_held_blocks(batch.block_ids)
        if batch.success:
            self._report_store_outcome(len(batch.keys), 0)
        else:
            self._stats["evict_lost"] += len(batch.keys)
            self._report_store_outcome(0, len(batch.keys))
            logger.warning(
                "OFFLOAD-EVICT-LOST keys=%d released_slots=%d lost_total=%d",
                len(batch.keys),
                released,
                self._stats["evict_lost"],
            )

    def _bump(self, name: str) -> None:
        """Count a lookup outcome and log the running totals every 10 s."""
        if not _TIER_STATS:
            return
        self._stats[name] += 1
        now = time.monotonic()
        if now - self._stats_last >= 10.0:
            self._stats_last = now
            st = self._stats
            logger.info(
                "TIER-STATS primary_hit=%d primary_inflight=%d sec_lookup=%d "
                "sec_hit=%d promo_ok=%d promo_fail=%d miss=%d store_failures=%d "
                "evict_cascaded=%d evict_lost=%d",
                st["primary_hit"],
                st["primary_inflight"],
                st["sec_lookup"],
                st["sec_hit"],
                st["promo_ok"],
                st["promo_fail"],
                st["miss"],
                st["store_failures"],
                st["evict_cascaded"],
                st["evict_lost"],
            )

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Check whether a single block is offloaded and ready.

        Algorithm:
            1. Process any completed async jobs first.
            2. Query primary tier — short-circuit on hit or in-flight.
            3. On primary miss, query secondary tiers — stop on first
               hit and initiate promotion.

        Args:
            key: Block hash to look up.
            req_context: Per-request context.

        Returns:
            True  — block is ready in the primary tier.
            None  — block found but not yet ready (primary in-flight,
                    promotion started, or a secondary tier is busy).
            False — block not found in any tier, or primary is full
                    and cannot accept a promotion.
        """
        self._maybe_process_finished_jobs()

        primary_hit = self.primary_tier.lookup(key, req_context)
        if primary_hit is True:
            self._bump("primary_hit")
            if self._hit_sink is not None:
                self._hit_sink("ram")
            return True
        if primary_hit is None:
            self._bump("primary_inflight")
            return None

        any_none = False
        for tier in self.secondary_tiers:
            self._bump("sec_lookup")
            result = tier.lookup(key, req_context)
            if result is True:
                self._bump("sec_hit")
                if not self._initiate_promotion(tier, key, req_context):
                    self._bump("promo_fail")
                    return False  # primary full, block unavailable
                self._bump("promo_ok")
                # Answered by the slower tier: this is the hit an aggregate
                # external-hit counter hides when the FS tier is idle.
                if self._hit_sink is not None:
                    self._hit_sink("fs")
                return None  # promotion started, retry later
            if result is None:
                any_none = True

        if any_none:
            return None
        self._bump("miss")
        return False

    def _initiate_promotion(
        self,
        tier: SecondaryTierManager,
        key: OffloadKey,
        req_context: ReqContext,
    ) -> bool:
        """
        Queue a block for promotion from a secondary tier to the primary tier.

        Allocates space in the primary tier immediately (sets ref_cnt=-1 so
        subsequent lookups within the same step see the slot as in-flight),
        then defers the actual submit_load() call to _flush_pending_promotions()
        so all blocks queued during one engine step are submitted as a single
        batched job.

        Args:
            tier: The secondary tier to promote from
            key: Block to promote
            req_context: Per-request context forwarded to primary.prepare_write().

        Returns:
            True if promotion was initiated, False if primary tier is full.
        """
        # Allocate space in primary tier for promoted block.
        # Must happen immediately so primary.lookup() returns None (in-flight)
        # for this key on any subsequent lookup() call within the same step,
        # preventing duplicate promotion attempts.
        if self.secondary_tiers and self.primary_tier.free_slots() <= 0:
            # Every slot is cached or held by an in-flight eviction copy. A
            # store attempt would evict a victim that cannot be reused until its
            # own copy lands, i.e. it would churn the pool without making room.
            # Report a miss; a later step retries once a copy finishes.
            return False
        primary_write_result = self.primary_tier.prepare_write([key], req_context)

        if primary_write_result is None:
            # Primary tier is full; caller should treat the block as unavailable
            # rather than retrying indefinitely.
            return False

        if not primary_write_result.keys_to_store:
            # The pool reported no refusal but had no room to allocate (its
            # slots are held by in-flight eviction copies), so no destination
            # block exists yet. Treat it as a miss and let a later step retry.
            return False

        store_spec = primary_write_result.store_spec
        assert isinstance(store_spec, CPULoadStoreSpec)
        # Defer submit_load to take_events(). Group by (tier, request) so each
        # request's blocks are submitted as one batched job per tier.
        tier_pending = self._pending_load_submissions.setdefault(tier, {})
        ctx_id = req_context.req_id
        if ctx_id not in tier_pending:
            tier_pending[ctx_id] = PendingPromotion(
                keys=[], block_ids=[], req_context=req_context
            )
        entry = tier_pending[ctx_id]
        entry.keys.extend(primary_write_result.keys_to_store)
        entry.block_ids.extend(store_spec.block_ids)
        return True

    def _flush_pending_promotions(self) -> None:
        """Submit one batched submit_load() per (tier, request).

        Called from take_events() at the end of each engine step, flushing
        all promotion requests deferred during lookup().
        """
        if not self._pending_load_submissions:
            return

        for tier, pending_by_ctx in self._pending_load_submissions.items():
            for entry in pending_by_ctx.values():
                job_id = self._next_job_id()
                job_metadata = JobMetadata(
                    job_id=job_id,
                    keys=entry.keys,
                    block_ids=np.array(entry.block_ids, dtype=np.int64),
                    is_promotion=True,
                    req_context=entry.req_context,
                )
                self._transfer_jobs[job_id] = job_metadata
                tier.submit_load(job_metadata)

        self._pending_load_submissions.clear()

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        """
        Prepare blocks to be loaded from primary tier to GPU.

        CRITICAL: This method calls _maybe_process_finished_jobs() FIRST to ensure
        that any completed promotions have been finalized and blocks are ready.

        This increments ref_cnt on the blocks in the primary tier, protecting
        them from eviction during the transfer.

        Args:
            keys: Blocks to prepare for loading.
            req_context: Per-request context.

        Returns:
            LoadStoreSpec for reading from primary tier.
        """
        # Process completed promotions to ensure blocks are ready
        self._maybe_process_finished_jobs()

        return self.primary_tier.prepare_load(keys, req_context)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as recently used in all tiers.

        Args:
            keys: Blocks to mark as recently used.
            req_context: Per-request context.
        """
        self.primary_tier.touch(keys, req_context)
        for tier in self.secondary_tiers:
            tier.touch(keys, req_context)

    def demote(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """Demote in the CPU tier; secondary tiers keep their own policy."""
        self.primary_tier.demote(keys, req_context)

    def pool_stats(self) -> tuple[int, int, int, int, int, int]:
        """Slot accounting of the primary pool, delegated to the tier that owns it."""
        return self.primary_tier.pool_stats()

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        """
        Mark blocks as done loading from primary tier to GPU.

        This decrements ref_cnt on the blocks in the primary tier, allowing
        them to be evicted again.

        Args:
            keys: Blocks that finished loading.
            req_context: Per-request context.
        """
        self.primary_tier.complete_load(keys, req_context)

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """
        Prepare blocks to be stored from GPU to primary tier.

        CRITICAL: This method calls _maybe_process_finished_jobs() FIRST to ensure
        that any completed async transfers have their ref_cnt decremented
        before the primary tier makes eviction decisions.

        For request-level tiers, blocks already present in the primary tier
        are immediately cascaded via submit_store().

        Args:
            keys: Blocks to prepare for storing.
            req_context: Per-request context.

        Returns:
            PrepareStoreOutput describing where to store blocks and what was
            evicted, or None if store cannot proceed.
        """
        # Step 1: Poll for completed async jobs FIRST
        # This decrements ref_cnt on primary blocks that have been
        # successfully transferred to secondary tiers.
        self._maybe_process_finished_jobs()

        # Step 2: Store to primary tier (new blocks only).
        # Cascading of these newly-stored blocks to ALL secondary tiers
        # happens later in complete_store(), after the GPU→Primary transfer
        # completes.
        primary_result = self.primary_tier.prepare_store(keys, req_context)

        if primary_result is None:
            return None

        # Step 3: For request-level tiers, cascade blocks already in primary
        request_level_tiers = self._request_level_tiers.get(req_context.req_id)
        if request_level_tiers is not None:
            keys_to_store_set = set(primary_result.keys_to_store)
            keys_already_in_primary = tuple(
                k for k in keys if k not in keys_to_store_set
            )
            if keys_already_in_primary:
                self._cascade_existing_blocks_to_request_level_tiers(
                    keys_already_in_primary, req_context, request_level_tiers
                )

        return primary_result

    def _cascade_existing_blocks_to_request_level_tiers(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext,
        request_level_tiers: set[SecondaryTierManager],
    ) -> None:
        """
        For tiers that requested request-level policy, submit_store() for
        blocks that are already present in the primary tier.
        """
        # Filter out keys that are not ready in primary (e.g. in-flight)
        ready_keys = tuple(
            k for k in keys if self.primary_tier.lookup(k, req_context) is True
        )
        if not ready_keys:
            return

        for tier in request_level_tiers:
            primary_blocks_spec = self.primary_tier.prepare_read(
                ready_keys, req_context
            )

            job_id = self._next_job_id()
            assert isinstance(primary_blocks_spec, CPULoadStoreSpec)
            job_metadata = JobMetadata(
                job_id=job_id,
                keys=ready_keys,
                block_ids=primary_blocks_spec.block_ids,
                is_promotion=False,
                req_context=req_context,
            )
            self._transfer_jobs[job_id] = job_metadata
            tier.submit_store(job_metadata)

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ):
        """
        Mark blocks as done storing from GPU to primary tier.

        This is where secondary tier cascading happens — after blocks are
        confirmed to be in the primary tier, they are cascaded to ALL
        secondary tiers.

        For each secondary tier:
        1. Call primary.prepare_read() to get LoadStoreSpec AND increment
           ref_cnt (protecting blocks during async transfer)
        2. Call tier.submit_store() to start async transfer: primary→secondary
        3. Track the job in _store_jobs dictionary

        Args:
            keys: Blocks that finished storing.
            success: Whether the GPU→primary transfer succeeded.
            req_context: Per-request context forwarded to primary.prepare_read().
        """
        # Step 1: Complete store in primary tier (makes blocks loadable)
        self.primary_tier.complete_store(keys, req_context, success)

        if not success:
            # The GPU→primary copy failed: the blocks are dropped from the pool
            # and can never reach a secondary tier. Count them as lost instead of
            # letting them vanish from the accounting.
            if keys and self.secondary_tiers:
                self._stats["evict_lost"] += len(keys)
                self._report_store_outcome(0, len(keys))
                logger.warning(
                    "OFFLOAD-STORE-DROPPED keys=%d lost_total=%d",
                    len(keys),
                    self._stats["evict_lost"],
                )
            return

        # Step 2: Cascade to ALL secondary tiers
        # For each secondary tier, call primary.prepare_read() to get the
        # LoadStoreSpec AND to increment ref_cnt (protecting blocks from
        # eviction during the async transfer). One prepare_read() call per
        # secondary tier.
        for tier in self.secondary_tiers:
            primary_blocks_spec = self.primary_tier.prepare_read(keys, req_context)

            # Submit async store job: primary→secondary
            job_id = self._next_job_id()

            # Track this store job
            assert isinstance(primary_blocks_spec, CPULoadStoreSpec)
            job_metadata = JobMetadata(
                job_id=job_id,
                keys=keys,
                block_ids=primary_blocks_spec.block_ids,
                is_promotion=False,
                req_context=req_context,
            )
            self._transfer_jobs[job_id] = job_metadata

            tier.submit_store(job_metadata)

        # Note: The async transfers are now in flight. Their completion is
        # tracked via get_finished_jobs() / _maybe_process_finished_jobs().

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """
        Query each secondary tier for its offload policy preference.

        Returns REQUEST_LEVEL if ANY secondary tier wants request-level.
        Only stores REQUEST_LEVEL tier decisions for use in prepare_store.
        """
        for tier in self.secondary_tiers:
            tier_ctx = tier.on_new_request(req_context)
            if tier_ctx.policy == OffloadPolicy.REQUEST_LEVEL:
                self._request_level_tiers[req_context.req_id].add(tier)

        policy = (
            OffloadPolicy.REQUEST_LEVEL
            if req_context.req_id in self._request_level_tiers
            else OffloadPolicy.BLOCK_LEVEL
        )
        return RequestOffloadingContext(policy=policy)

    def on_request_finished(self, req_context: ReqContext) -> None:
        self.primary_tier.on_request_finished(req_context)
        for tier in self.secondary_tiers:
            tier.on_request_finished(req_context)
        self._request_level_tiers.pop(req_context.req_id, None)

    def take_events(self) -> Iterable[OffloadingEvent]:
        """
        End-of-step hook: flush deferred work, yield events, reset per-step state.

        Called once per engine step from Scheduler.update_from_output() →
        connector.take_events(). Ensures _maybe_process_finished_jobs() has run
        at least once this step, flushes pending promotions, yields collected
        events, and resets the per-step flag.

        Yields:
            New OffloadingEvents collected since the last call.
        """
        # TODO: Move _flush_pending_promotions() to a dedicated end_of_batch()
        # hook once one exists. For now, take_events() serves as the flush
        # point under the assumption that it is called at the end of each
        # engine step (Scheduler.update_from_output() → connector.take_events()).
        # When the dedicated hook is added, update tests that rely on
        # take_events() to signal end of step.

        self._maybe_process_finished_jobs()

        self._flush_pending_promotions()

        # Reset the per-step gate so next step's first call does real work.
        self._processed_jobs_this_step = False

        if self.events is not None:
            yield from self.events
            self.events.clear()

        yield from self.primary_tier.take_events()

    def shutdown(self) -> None:
        """Shutdown all tiers and release resources."""
        for tier in self.secondary_tiers:
            tier.shutdown()
        self.primary_tier.shutdown()
