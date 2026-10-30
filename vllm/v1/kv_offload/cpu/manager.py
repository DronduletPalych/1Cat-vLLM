# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections import OrderedDict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Literal

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    OffloadPolicy,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

from vllm.logger import init_logger

logger = init_logger(__name__)

# A cascade handler receives the blocks the replacement policy just evicted and
# returns the keys whose slot it now owns: their data is copied elsewhere before
# the slot may be reused. Every other victim's slot is freed immediately.
EvictionCascade = Callable[
    [Sequence[tuple[OffloadKey, BlockStatus]], ReqContext], Collection[OffloadKey]
]

# A reclaim handler polls the tier below for the copies started by an eviction
# cascade, finishing them so the slots they pin come back to the pool. It gets
# the number of slots the caller needs and a time budget in seconds, and returns
# how many slots were returned. Waiting on it is what lets a full pool accept a
# new block in the same call instead of refusing it.
ReclaimHeldSlots = Callable[[int, float], int]

_CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
}


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy (LRU or ARC).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
        share_pages: bool = False,
        defer_inflight_lookup: bool = False,
        group_label: str = "-",
        partial_admit: bool = False,
    ):
        self.medium: str = CPULoadStoreSpec.medium()
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = _CACHE_POLICIES.get(cache_policy)
        if policy_cls is None:
            raise ValueError(
                f"Unknown cache policy: {cache_policy!r}. "
                f"Supported: {list(_CACHE_POLICIES)}"
            )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)
        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size

        # Number of block references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )
        # CPU block id -> every group key stored in that page. One page holds
        # all KV groups of a token block, so several keys share a BlockStatus.
        # Grouped pools address disjoint per-group tensors and never share one.
        self.share_pages: bool = share_pages
        # A tier with an asynchronous secondary tier answers ``None`` so the
        # scheduler retries once the promotion lands. Without one, an
        # in-flight write must be a miss: parking the request would leave an
        # idle engine unable to complete the write it is waiting on.
        self.defer_inflight_lookup: bool = defer_inflight_lookup
        self._keys_by_block: dict[int, set[OffloadKey]] = {}
        # Store-admission diagnostics: label of the KV group this pool serves
        # (empty for a shared-page pool) and how often the pool could not make
        # room for a store.
        self.group_label: str = group_label
        # A grouped pool stores behind a fixed number of private slots. When the
        # batch does not fit, admitting its leading part and letting the caller
        # retry the tail keeps blocks flowing to the secondary tiers; a shared
        # page pool keeps the all-or-nothing contract.
        self.partial_admit: bool = partial_admit
        self._admit_last_log: float = 0.0
        # Slots whose data is being copied out of the pool before reuse (see
        # set_eviction_cascade). They are out of the policy and out of the free
        # list: the slot cannot be reallocated until release_held_blocks().
        self._eviction_cascade: EvictionCascade | None = None
        self._eviction_reclaim: ReclaimHeldSlots | None = None
        self._held_block_ids: set[int] = set()
        self._cascaded_total: int = 0
        # How long an admission may spend waiting for the copies of the slots it
        # just evicted. A slot is not reusable until its copy lands, so waiting
        # for the copy is what makes "pool full" an admission instead of a
        # refusal. 0 disables the wait (the pool then refuses and the caller
        # retries on a later step).
        self.reclaim_timeout: float = (
            float(os.environ.get("VLLM_EVICT_RECLAIM_TIMEOUT_MS", "250")) / 1000.0
        )
        self._reclaim_waits: int = 0
        self._reclaim_wait_ms: int = 0
        self._reclaim_timeouts: int = 0
        # Blocks the pool could not place at all. `blocks`/`lost` count every
        # refused key (an offer that produced no slot and no copy), so a refusal
        # is visible in the loss accounting instead of hiding behind lost=0.
        self._refusals: dict[str, int] = {"batches": 0, "blocks": 0, "lost": 0}

    # --- eviction cascade ---

    def set_eviction_cascade(self, cascade: EvictionCascade) -> None:
        """Install the handler that preserves victims before their slot is freed.

        Without one, an evicted block is simply dropped (the CPU tier is the
        last tier). With one, the policy's victims are handed to the handler
        first; the keys it returns keep their slot until
        release_held_blocks() puts it back on the free list.
        """
        self._eviction_cascade = cascade

    def set_eviction_reclaim(self, reclaim: ReclaimHeldSlots) -> None:
        """Install the handler that finishes in-flight eviction copies.

        Without one, a slot held by a cascade is only released when the caller
        comes back: a full pool then refuses the new block and the caller has to
        retry. With one, the admission waits (bounded by ``reclaim_timeout``)
        for the copy it just started, so the block it wanted to store is
        accepted in the same call.
        """
        self._eviction_reclaim = reclaim

    def _reclaim_held(self, wanted: int) -> int:
        """Wait, bounded, for slots pinned by in-flight eviction copies.

        Returns the number of slots that came back. A no-op when nothing is
        held or no reclaim handler is installed, so a pool without a secondary
        tier keeps its plain LRU behavior.
        """
        if wanted <= 0 or not self._held_block_ids:
            return 0
        if self._eviction_reclaim is None or self.reclaim_timeout <= 0:
            return 0
        held_before = len(self._held_block_ids)
        start = time.monotonic()
        try:
            self._eviction_reclaim(wanted, self.reclaim_timeout)
        except Exception:  # never let a broken tier abort an admission
            logger.exception(
                "FS-RECLAIM-ERROR grp=%s wanted=%d held=%d",
                self.group_label,
                wanted,
                held_before,
            )
            return 0
        finally:
            elapsed = time.monotonic() - start
            self._reclaim_waits += 1
            self._reclaim_wait_ms += int(elapsed * 1000.0)
            if len(self._held_block_ids) >= held_before:
                self._reclaim_timeouts += 1
        return held_before - len(self._held_block_ids)

    def refusal_stats(self) -> dict[str, int]:
        """Store refusals so far: (batches, blocks, blocks counted as lost)."""
        return dict(self._refusals)

    def _release_evicted(
        self,
        evicted: Sequence[tuple[OffloadKey, BlockStatus]],
        req_context: ReqContext,
    ) -> tuple[list[tuple[OffloadKey, BlockStatus]], list[tuple[OffloadKey, BlockStatus]]]:
        """Split policy victims into (slot freed now, slot held by the cascade).

        The policy has already unlinked every victim, so a held block is no
        longer visible to lookup()/prepare_load(); only its memory stays alive,
        which is exactly what a secondary tier needs to read it.
        """
        held_keys: set[OffloadKey] = set()
        if self._eviction_cascade is not None and evicted:
            held_keys = set(self._eviction_cascade(evicted, req_context))
        freed: list[tuple[OffloadKey, BlockStatus]] = []
        held: list[tuple[OffloadKey, BlockStatus]] = []
        for key, block in evicted:
            if key in held_keys:
                self._held_block_ids.add(int(block.block_id))
                held.append((key, block))
                continue
            self._free_block(block)
            freed.append((key, block))
        if held:
            self._cascaded_total += len(held)
        return freed, held

    def release_held_blocks(self, block_ids: Iterable[int]) -> int:
        """Return slots held by an eviction cascade to the free list.

        Idempotent: an id is released once, and only if it is actually held, so
        a late completion after reset_cache() cannot free a foreign slot.
        """
        released = 0
        for block_id in block_ids:
            block_id = int(block_id)
            if block_id in self._held_block_ids:
                self._held_block_ids.discard(block_id)
                self._free_list.append(block_id)
                released += 1
        return released

    def held_blocks(self) -> int:
        """Slots currently pinned by an in-flight eviction copy."""
        return len(self._held_block_ids)

    def free_slots(self) -> int:
        """Slots that can be allocated right now (see _get_num_free_blocks)."""
        return self._get_num_free_blocks()

    # --- block pool ---

    def _get_num_free_blocks(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    def _admit_composition(self, protected: set[OffloadKey]) -> tuple[int, int, int]:
        """Count cached blocks by why they cannot be evicted (diagnostics only).

        Returns (evictable, pinned, not_ready): ``pinned`` blocks are in use as
        a transfer source (or not yet ready), ``not_ready`` are slots reserved
        by an in-flight store/promotion, ``evictable`` are the LRU candidates
        that are neither pinned nor protected by this store call.
        """
        evictable = pinned = not_ready = 0
        for key, block in self._policy.iter_blocks():
            if block.ref_cnt > 0:
                pinned += 1
            elif block.ref_cnt < 0:
                not_ready += 1
            elif key not in protected:
                evictable += 1
        return evictable, pinned, not_ready

    def _log_admit_refusal(
        self,
        new: int,
        need: int,
        protected: set[OffloadKey],
        reason: str,
        admitted: int = 0,
    ) -> None:
        """Record a store batch the pool cannot fully admit, with its composition."""
        unserved = max(0, new - admitted)
        self._refusals["batches"] += 1
        self._refusals["blocks"] += unserved
        # A key without a host slot never reaches a secondary tier, so a refused
        # key is a lost block unless the caller manages to offer it again. It is
        # counted here, once per offer, instead of vanishing into lost=0.
        self._refusals["lost"] += unserved
        now = time.monotonic()
        if now - self._admit_last_log < 2.0:
            return
        self._admit_last_log = now
        evictable, pinned, not_ready = self._admit_composition(protected)
        logger.info(
            "FS-ADMIT-REFUSE grp=%s reason=%s new=%d admitted=%d unserved=%d free=%d "
            "pool=%d alloc=%d need_evict=%d evictable=%d pinned=%d not_ready=%d "
            "protected=%d held=%d refusals=%d refused_blocks=%d waits=%d "
            "wait_ms=%d wait_timeouts=%d",
            self.group_label,
            reason,
            new,
            admitted,
            unserved,
            self._get_num_free_blocks(),
            self._num_blocks,
            self._num_allocated_blocks,
            need,
            evictable,
            pinned,
            not_ready,
            len(protected),
            len(self._held_block_ids),
            self._refusals["batches"],
            self._refusals["blocks"],
            self._reclaim_waits,
            self._reclaim_wait_ms,
            self._reclaim_timeouts,
        )

    def reclaim_stats(self) -> tuple[int, int, int]:
        """Bounded eviction-copy waits so far: (calls, total_ms, timeouts)."""
        return (self._reclaim_waits, self._reclaim_wait_ms, self._reclaim_timeouts)

    def pool_stats(self) -> tuple[int, int, int, int, int, int]:
        """(pool, allocated, ready, pinned, not_ready, held) counts, diagnostics only."""
        ready, pinned, not_ready = self._admit_composition(set())
        return (
            self._num_blocks,
            self._num_allocated_blocks,
            ready,
            pinned,
            not_ready,
            len(self._held_block_ids),
        )

    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        num_fresh = min(len(keys), self._num_blocks - self._num_allocated_blocks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])

    # --- OffloadingManager interface ---

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        if self.counts is not None:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)
                self.counts[key] = 1
        block = self._policy.get(key)
        if block is None:
            return False
        if not block.is_ready:
            # A store for this page is still in flight. Returning None asks the
            # scheduler to defer the request to its skipped-waiting queue, which
            # is only revisited on engine steps. An engine with no running
            # requests never takes a step, so it never polls the worker for the
            # completed transfer, the page never becomes ready, and the request
            # never wakes: a permanent stall with the KV pool idle. Treat the
            # in-flight page as a miss instead; the caller uses the shorter
            # prefix it already has and hits this page on a later request.
            return None if self.defer_inflight_lookup else False
        return True

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        blocks = []
        pinned: set[int | tuple[int, int]] = set()
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            page = (
                block.block_id
                if self.share_pages
                else (get_offload_group_idx(key), block.block_id)
            )
            if page not in pinned:
                block.ref_cnt += 1
                pinned.add(page)
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    def contains(self, key: OffloadKey) -> bool:
        """Read-only membership test. Diagnostics only: no LRU touch."""
        return self._policy.get(key) is not None

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys)

    def demote(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.demote(keys)

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        released: set[int | tuple[int, int]] = set()
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            page = (
                block.block_id
                if self.share_pages
                else (get_offload_group_idx(key), block.block_id)
            )
            if page in released:
                continue
            released.add(page)
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        if self.counts is not None:
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]

        if not self.share_pages:
            return self._prepare_store_per_key(keys, req_context)

        # A CPU page holds every KV group of one token block. Group the keys by
        # block hash so all groups of a block share one page and one block id.
        by_hash: dict[bytes, list[OffloadKey]] = {}
        for key in keys:
            by_hash.setdefault(get_offload_block_hash(key), []).append(key)

        attached: set[OffloadKey] = set()
        new_hashes: list[bytes] = []
        for block_hash, group_keys in by_hash.items():
            block = next(
                (
                    self._policy.get(k)
                    for k in group_keys
                    if self._policy.get(k) is not None
                ),
                None,
            )
            if block is None:
                new_hashes.append(block_hash)
                continue
            for k in group_keys:
                if self._policy.get(k) is None:
                    self._policy.insert(k, block)
                    self._keys_by_block.setdefault(block.block_id, set()).add(k)
                    attached.add(k)

        if not new_hashes and not attached:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        num_blocks_to_evict = len(new_hashes) - self._get_num_free_blocks()
        if num_blocks_to_evict > 0 and self._held_block_ids:
            # A slot whose copy is already in flight comes back without evicting
            # anything else: collect it before choosing victims.
            self._reclaim_held(num_blocks_to_evict)
            num_blocks_to_evict = len(new_hashes) - self._get_num_free_blocks()

        to_evict: list[OffloadKey] = []
        cascaded: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                # A shared-page pool keeps its all-or-nothing contract: one page
                # holds every KV group of a token block, so a partial batch would
                # have to be re-derived per page. Only the per-key (grouped) pools
                # admit partially; this layout is not used with Mamba hybrids.
                self._log_admit_refusal(
                    len(new_hashes),
                    num_blocks_to_evict,
                    protected,
                    "shared-page pool full",
                )
                return None
            for key, block in evicted:
                for k in list(self._keys_by_block.pop(block.block_id, {key})):
                    if k != key and self._policy.get(k) is not None:
                        self._policy.remove(k)
            freed, held = self._release_evicted(evicted, req_context)
            to_evict = [key for key, _ in freed]
            cascaded = [key for key, _ in held]
            if cascaded and self._get_num_free_blocks() < len(new_hashes):
                # The victims' slots come back as soon as their copies land, so
                # wait for the copies this eviction started instead of refusing.
                self._reclaim_held(len(new_hashes) - self._get_num_free_blocks())
            if cascaded and self._get_num_free_blocks() < len(new_hashes):
                # Victims are being copied out and their slots are not free yet.
                # Refusing keeps the all-or-nothing contract; the refusal is
                # counted as lost so the channel stays visible.
                self._log_admit_refusal(
                    len(new_hashes),
                    num_blocks_to_evict,
                    protected,
                    "shared-page victims held by eviction cascade",
                    len(freed),
                )
                return None

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks([OffloadKey(h) for h in new_hashes])
        assert len(blocks) == len(new_hashes), (
            "Block pool did not allocate the expected number of blocks"
        )

        block_by_key: dict[OffloadKey, BlockStatus] = {}
        for block_hash, block in zip(new_hashes, blocks):
            for key in by_hash[block_hash]:
                self._policy.insert(key, block)
                self._keys_by_block.setdefault(block.block_id, set()).add(key)
                block_by_key[key] = block

        for key in attached:
            block = self._policy.get(key)
            if block is not None:
                block_by_key[key] = block

        # Keep the caller's key order: the scheduler builds the source spec in
        # that order and the two sides must line up block for block.
        store_set = set(block_by_key)
        keys_to_store = [k for k in keys if k in store_set]
        store_spec = self._get_load_store_spec(
            keys_to_store, [block_by_key[k] for k in keys_to_store]
        )

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
            cascaded_keys=tuple(cascaded),
        )

    def _prepare_store_per_key(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        # Grouped pools give every key its own slot: block ids address disjoint
        # per-group tensors, so keys must never be merged by block hash.
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        # A slot this pool evicted is not free until the copy that persists it
        # lands. Collect those slots first: a copy already in flight returns its
        # slot without evicting anything, and each new offer that evicts one more
        # victim for the same admission only churns the pool.
        free = self._get_num_free_blocks()
        if len(keys_to_store) > free and self._held_block_ids:
            self._reclaim_held(len(keys_to_store) - free)
            free = self._get_num_free_blocks()

        deferred: tuple[OffloadKey, ...] = ()
        to_evict: list[OffloadKey] = []
        cascaded: list[OffloadKey] = []
        need = len(keys_to_store) - free
        if need > 0:
            protected = set(keys)
            evictable, _, _ = self._admit_composition(protected)
            if free == 0 and self._held_block_ids:
                # No reusable slot anywhere and the copies that own the pinned
                # ones did not land inside the budget. Evicting one more victim
                # would pin a second slot for this same admission, so report the
                # refusal (counted as lost) instead of churning the pool.
                self._log_admit_refusal(
                    len(keys_to_store),
                    need,
                    protected,
                    "slots pinned by eviction copies past the budget",
                    0,
                )
                return None
            want = min(need, evictable)
            if want < need and not self.partial_admit:
                self._log_admit_refusal(
                    len(keys_to_store),
                    need,
                    protected,
                    "per-key pool full",
                    0,
                )
                return None
            if want <= 0 and free == 0:
                # Nothing evictable and no free slot: a refusal is the only
                # answer, and it is counted as a lost block.
                self._log_admit_refusal(
                    len(keys_to_store),
                    need,
                    protected,
                    "per-key pool full",
                    0,
                )
                return None
            if want > 0:
                # Evict, but never more victims than the pool has: an
                # over-large request makes the policy refuse atomically, and
                # that refusal would drop the whole batch.
                evicted = self._policy.evict(want, protected)
                # Every victim is either freed now or handed to the eviction
                # cascade, which copies it to the slower tier before the slot is
                # reusable.
                freed, held = self._release_evicted(evicted or [], req_context)
                to_evict = [key for key, _ in freed]
                cascaded = [key for key, _ in held]

        # Slots held by the cascade are not free until their copy lands. Wait for
        # those copies instead of refusing: the block the caller wants to store
        # was accepted for exactly this reason.
        free = self._get_num_free_blocks()
        if free < len(keys_to_store) and self._held_block_ids:
            self._reclaim_held(len(keys_to_store) - free)
            free = self._get_num_free_blocks()

        if free == 0 and keys_to_store:
            # Every slot is cached, pinned by a transfer, or held by an eviction
            # copy that missed the budget. Refuse the batch (and count it) instead
            # of returning an empty store: the caller retries, while an empty
            # keys_to_store would tell the scheduler there is nothing left to
            # store and the blocks would never be offered again.
            self._log_admit_refusal(
                len(keys_to_store) + len(deferred),
                len(keys_to_store),
                set(keys),
                "all slots held by eviction copies",
                0,
            )
            return None
        if free < len(keys_to_store):
            deferred = tuple(keys_to_store[free:]) + deferred
            keys_to_store = keys_to_store[:free]

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)

        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
            deferred_keys=deferred,
            cascaded_keys=tuple(cascaded),
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        stored_keys: list[OffloadKey] = []
        seen: set[int | tuple[int, int]] = set()

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is None:
                    continue
                page = (
                    block.block_id
                    if self.share_pages
                    else (get_offload_group_idx(key), block.block_id)
                )
                if page in seen:
                    continue
                seen.add(page)
                if not block.is_ready:
                    block.ref_cnt = 0
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is None:
                    continue
                page = (
                    block.block_id
                    if self.share_pages
                    else (get_offload_group_idx(key), block.block_id)
                )
                if page in seen:
                    continue
                seen.add(page)
                if block.is_ready:
                    continue
                if not self.share_pages:
                    self._policy.remove(key)
                    self._free_block(block)
                    continue
                for k in list(self._keys_by_block.pop(block.block_id, ())):
                    if self._policy.get(k) is not None:
                        self._policy.remove(k)
                self._free_block(block)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    def reset_cache(self) -> None:
        # Clear ALL blocks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused offload block IDs.
        self._policy.clear()
        self._keys_by_block.clear()

        self._free_list.clear()
        self._num_allocated_blocks = 0
        # A reset invalidates every slot, including those an eviction cascade
        # still holds; a late completion must not free a slot of the new pool.
        self._held_block_ids.clear()

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()


class GroupedCPUOffloadingManager(OffloadingManager):
    """Independent group pools whose block IDs address disjoint CPU tensors.

    A key only stores one KV group. Giving each group its own pool avoids
    charging small groups for the largest shared GPU tensor layout. IDs may
    repeat across groups; the GPU transfer spec carries the original group ID.
    """

    def __init__(
        self,
        managers: Mapping[int, OffloadingManager],
        mamba_groups: Collection[int] | None = None,
    ):
        self.managers = managers
        # Groups backed by Mamba state. A block is only restorable when its
        # state half is present: the prefix-cache hit length is the minimum
        # across groups, so attention KV past the last state checkpoint is
        # never reused and only evicts restorable blocks.
        self._mamba_groups = frozenset(mamba_groups or ())
        self._dropped_stateless_keys = 0
        self._gate_info: dict = {}
        self._stats_enabled = os.environ.get("VLLM_TIER_STATS") == "1"
        # Per group: [in, new, adm, def, out, lost, refused, casc, ref].
        #   in      — blocks offered to this group's prepare_store (a retried key
        #             is offered again and counted again)
        #   new     — offered blocks the pool did not already hold, i.e. the ones
        #             that need a slot
        #   adm     — blocks admitted into the group's primary pool
        #   def     — blocks the pool deferred to the caller's next attempt
        #             (partial admit: no slot yet, retried, not lost)
        #   out     — admitted stores the secondary tier CONFIRMED as written
        #   lost    — blocks that left the pool without ever being persisted,
        #             including refused blocks (a refusal has no slot, so it can
        #             never reach a secondary tier)
        #   refused — blocks in batches the group refused outright
        #   casc    — eviction victims the secondary tier confirmed as written
        #   ref     — store batches the group refused
        self._stats: dict[int, list[int]] = {}
        self._stats_last = time.monotonic()
        # Boundary state offers the scheduler read from its requests before the
        # pool saw anything: [pre, not_ready, busy, retrycap].
        #   pre       — offers the scheduler tried to place in this interval
        #   not_ready — held back: the block's attention half is not
        #               materialised yet (state alone is never written)
        #   busy      — held back: the request's transfer or this pool was busy
        #   retrycap  — dropped before the pool: queue cap or an unstorable
        #               boundary
        # Without them a boundary that never reaches the pool is invisible: the
        # per-group `in` counter only sees what was offered to prepare_store.
        self._boundary_stats: list[int] = [0, 0, 0, 0]
        # Per group: [ram_hit_lookups, fs_hit_lookups]. Hits answered by the
        # primary pool and hits answered by a secondary tier are counted apart:
        # an aggregate external-hit number hides a tier that does nothing.
        self._hit_blocks: dict[int, list[int]] = {}
        self._hit_last = time.monotonic()
        self._hit_taken: dict[int, tuple[int, int]] = {}
        for group_idx, manager in managers.items():
            store_sink = getattr(manager, "set_store_outcome_sink", None)
            if store_sink is not None:
                store_sink(
                    lambda confirmed, lost, g=group_idx: self._on_store_outcome(
                        g, confirmed, lost
                    )
                )
            cascade_sink = getattr(manager, "set_cascade_outcome_sink", None)
            if cascade_sink is not None:
                cascade_sink(
                    lambda confirmed, lost, g=group_idx: self._on_cascade_outcome(
                        g, confirmed, lost
                    )
                )
            hit_sink = getattr(manager, "set_hit_sink", None)
            if hit_sink is not None:
                hit_sink(lambda tier, g=group_idx: self._on_hit(g, tier))

    def _on_store_outcome(self, group_idx: int, confirmed: int, lost: int) -> None:
        """Count admitted stores a secondary tier confirmed (or dropped) per group."""
        if confirmed:
            self._stats.setdefault(group_idx, [0] * 9)[4] += confirmed
        if lost:
            self._stats.setdefault(group_idx, [0] * 9)[5] += lost
        self._maybe_log_stats()

    def _on_cascade_outcome(self, group_idx: int, confirmed: int, lost: int) -> None:
        """Count eviction copies a secondary tier confirmed (or dropped) per group.

        Kept apart from the store outcome: a cascade writes a block the pool had
        already admitted, so counting it as `out` made `out` exceed `adm` and
        hid stores that were written without ever being admitted.
        """
        if confirmed:
            self._stats.setdefault(group_idx, [0] * 9)[7] += confirmed
        if lost:
            self._stats.setdefault(group_idx, [0] * 9)[5] += lost
        self._maybe_log_stats()

    def _on_hit(self, group_idx: int, tier: str) -> None:
        acc = self._hit_blocks.setdefault(group_idx, [0, 0])
        acc[0 if tier == "ram" else 1] += 1
        self._maybe_log_hits()

    def take_hit_stats(self) -> dict[int, tuple[int, int]]:
        """Per-group (ram, fs) hit-lookup counts since the previous call."""
        deltas: dict[int, tuple[int, int]] = {}
        for group_idx, (ram, fs) in self._hit_blocks.items():
            prev_ram, prev_fs = self._hit_taken.get(group_idx, (0, 0))
            if ram != prev_ram or fs != prev_fs:
                deltas[group_idx] = (ram - prev_ram, fs - prev_fs)
                self._hit_taken[group_idx] = (ram, fs)
        return deltas

    def _bump_group(self, group_idx: int, slot: int, value: int) -> None:
        if not self._stats_enabled:
            return
        self._stats.setdefault(group_idx, [0] * 9)[slot] += value
        self._maybe_log_stats()

    def note_boundary_attempt(
        self, pre: int = 0, not_ready: int = 0, busy: int = 0, retrycap: int = 0
    ) -> None:
        """Add the scheduler's boundary-state counters to this interval's log."""
        if not self._stats_enabled:
            return
        acc = self._boundary_stats
        acc[0] += pre
        acc[1] += not_ready
        acc[2] += busy
        acc[3] += retrycap
        self._maybe_log_stats()

    def _secondary_lookup_row(self) -> dict[int, tuple[int, int]]:
        """Per-group (secondary lookups, secondary hits), cumulative."""
        row: dict[int, tuple[int, int]] = {}
        for group_idx, manager in self.managers.items():
            stats = getattr(manager, "secondary_lookup_stats", None)
            if stats is None:
                continue
            row[group_idx] = stats()
        return row

    def _maybe_log_stats(self, force: bool = False) -> None:
        if not self._stats_enabled:
            return
        now = time.monotonic()
        if not force and now - self._stats_last < 10.0:
            return
        self._stats_last = now
        boundary = self._boundary_stats
        logger.info(
            "GROUPED-STORE in_new_adm_def_out_lost_refused_casc_ref=%s "
            "pre_skip_pre_notready_busy_retrycap=%s sec_lk_sec_hit=%s",
            {g: tuple(v) for g, v in sorted(self._stats.items())},
            tuple(boundary),
            self._secondary_lookup_row(),
        )
        # The counters are per interval: the pool reports what the scheduler
        # offered and held back since the previous line.
        for idx in range(len(boundary)):
            boundary[idx] = 0
        # (pool, allocated, ready, pinned, not_ready, held) per group:
        # distinguishes a pool that is full of in-flight transfers from one that
        # is full of live cache entries, is held by an eviction copy, or shows
        # whether a slot ever comes back.
        logger.info(
            "FS-POOL group_pool_alloc_ready_pinned_notready_held=%s",
            {
                g: m.pool_stats()  # type: ignore[attr-defined]
                for g, m in sorted(self.managers.items())
            },
        )
        logger.info(
            "FS-REFUSE group_ref_batches_ref_blocks_wait_calls_wait_ms_timeouts=%s",
            {g: self._refuse_row(m) for g, m in sorted(self.managers.items())},
        )

    @staticmethod
    def _refuse_row(manager: OffloadingManager) -> tuple[int, ...]:
        """(refused batches, refused blocks, reclaim waits, ms, timeouts)."""
        refusals = getattr(manager, "refusal_stats", None)
        reclaim = getattr(manager, "reclaim_stats", None)
        if refusals is None or reclaim is None:
            return (0, 0, 0, 0, 0)
        stats = refusals()
        return (stats["batches"], stats["blocks"]) + reclaim()

    def _maybe_log_hits(self, force: bool = False) -> None:
        if not self._stats_enabled:
            return
        now = time.monotonic()
        if not force and now - self._hit_last < 10.0:
            return
        self._hit_last = now
        ram_total = sum(v[0] for v in self._hit_blocks.values())
        fs_total = sum(v[1] for v in self._hit_blocks.values())
        logger.info(
            "GROUPED-HITS group_ram_fs_hit_blocks=%s ram_hits=%d fs_hits=%d",
            {g: tuple(v) for g, v in sorted(self._hit_blocks.items())},
            ram_total,
            fs_total,
        )

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        contexts = [m.on_new_request(req_context) for m in self.managers.values()]
        return RequestOffloadingContext(
            policy=(
                OffloadPolicy.REQUEST_LEVEL
                if any(c.policy == OffloadPolicy.REQUEST_LEVEL for c in contexts)
                else OffloadPolicy.BLOCK_LEVEL
            )
        )

    def on_request_finished(self, req_context: ReqContext) -> None:
        for manager in self.managers.values():
            manager.on_request_finished(req_context)

    def shutdown(self) -> None:
        for manager in self.managers.values():
            manager.shutdown()

    def _partition(self, keys: Collection[OffloadKey]) -> dict[int, list[OffloadKey]]:
        groups: dict[int, list[OffloadKey]] = {}
        for key in keys:
            group_idx = get_offload_group_idx(key)
            assert group_idx in self.managers
            groups.setdefault(group_idx, []).append(key)
        return groups

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        return self.managers[get_offload_group_idx(key)].lookup(key, req_context)

    def contains(self, key: OffloadKey) -> bool:
        manager = self.managers.get(get_offload_group_idx(key))
        return False if manager is None else manager.contains(key)

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        for group_idx, group_keys in self._partition(keys).items():
            self.managers[group_idx].touch(group_keys, req_context)

    def demote(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        for group_idx, group_keys in self._partition(keys).items():
            self.managers[group_idx].demote(group_keys, req_context)

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> CPULoadStoreSpec:
        block_ids: dict[OffloadKey, int] = {}
        for group_idx, group_keys in self._partition(keys).items():
            spec = self.managers[group_idx].prepare_load(group_keys, req_context)
            assert isinstance(spec, CPULoadStoreSpec)
            block_ids.update(zip(group_keys, map(int, spec.block_ids)))
        return CPULoadStoreSpec([block_ids[key] for key in keys])

    def complete_load(self, keys: Collection[OffloadKey], req_context: ReqContext):
        for group_idx, group_keys in self._partition(keys).items():
            self.managers[group_idx].complete_load(group_keys, req_context)

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        # No store gate: attention of a block below its Mamba checkpoint
        # boundary is still required to rebuild the prefix chain from block
        # zero. Dropping it made every tier lookup miss.
        block_ids: dict[OffloadKey, int] = {}
        evicted_keys: list[OffloadKey] = []
        cascaded_keys: list[OffloadKey] = []
        deferred_keys: list[OffloadKey] = []
        refused_groups: list[int] = []
        for group_idx, group_keys in self._partition(keys).items():
            self._bump_group(group_idx, 0, len(group_keys))
            manager = self.managers[group_idx]
            # Blocks the pool does not hold yet are the ones that need a slot:
            # `in` counts every offer (a retried key is offered again), `new` is
            # what the pool has to place.
            self._bump_group(
                group_idx,
                1,
                sum(0 if manager.contains(key) else 1 for key in group_keys),
            )
            output = manager.prepare_store(group_keys, req_context)
            if output is None:
                # This pool cannot take any of its keys right now. Keep the
                # other groups' admissions instead of cancelling the batch: a
                # block that never gets a host slot never reaches a secondary
                # tier either, so one starved group must not drop the blocks
                # that do fit. The caller retries the refused keys.
                refused_groups.append(group_idx)
                self._bump_group(group_idx, 8, 1)
                # A refused key has no slot and no copy, so it is a block at
                # risk of never being persisted: count it, or the channel stays
                # invisible behind lost=0.
                self._bump_group(group_idx, 6, len(group_keys))
                self._bump_group(group_idx, 5, len(group_keys))
                deferred_keys.extend(group_keys)
                continue
            assert isinstance(output.store_spec, CPULoadStoreSpec)
            self._bump_group(group_idx, 2, len(output.keys_to_store))
            self._bump_group(group_idx, 3, len(output.deferred_keys))
            block_ids.update(
                zip(output.keys_to_store, map(int, output.store_spec.block_ids))
            )
            evicted_keys.extend(output.evicted_keys)
            cascaded_keys.extend(output.cascaded_keys)
            deferred_keys.extend(output.deferred_keys)
        if refused_groups and not block_ids:
            # Nothing was admitted anywhere: report a refusal so the caller
            # retries the batch instead of being told every key is present.
            return None
        stored_set = set(block_ids)
        stored = [key for key in keys if key in stored_set]
        return PrepareStoreOutput(
            keys_to_store=stored,
            store_spec=CPULoadStoreSpec([block_ids[key] for key in stored]),
            evicted_keys=evicted_keys,
            deferred_keys=tuple(deferred_keys),
            cascaded_keys=tuple(cascaded_keys),
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        for group_idx, group_keys in self._partition(keys).items():
            self.managers[group_idx].complete_store(group_keys, req_context, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        for manager in self.managers.values():
            yield from manager.take_events()

    def reset_cache(self) -> None:
        for manager in self.managers.values():
            manager.reset_cache()