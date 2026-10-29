# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections import OrderedDict
from collections.abc import Collection, Iterable, Mapping
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

    # --- block pool ---

    def _get_num_free_blocks(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

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
            return self._prepare_store_per_key(keys)

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

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None
            for key, block in evicted:
                for k in list(self._keys_by_block.pop(block.block_id, {key})):
                    if k != key and self._policy.get(k) is not None:
                        self._policy.remove(k)
                self._free_block(block)
                to_evict.append(key)

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
        )

    def _prepare_store_per_key(
        self, keys: Collection[OffloadKey]
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

        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None
            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)

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
        self._stats: dict[int, list[int]] = {}
        self._stats_last = time.monotonic()

    def _bump_group(self, group_idx: int, slot: int, value: int) -> None:
        if not self._stats_enabled:
            return
        acc = self._stats.setdefault(group_idx, [0, 0, 0])
        acc[slot] += value
        now = time.monotonic()
        if now - self._stats_last >= 10.0:
            self._stats_last = now
            logger.info(
                "GROUPED-STORE group_in_out_ev=%s",
                {g: tuple(v) for g, v in sorted(self._stats.items())},
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
        for group_idx, group_keys in self._partition(keys).items():
            self._bump_group(group_idx, 0, len(group_keys))
            output = self.managers[group_idx].prepare_store(group_keys, req_context)
            if output is None:
                # The scheduler advances every group's store cursor on success.
                # Release only this call's new reservations so all groups retry.
                # Already performed evictions are not undone; pre-existing
                # in-flight stores must not be cancelled.
                self.complete_store(list(block_ids), req_context, success=False)
                return None
            assert isinstance(output.store_spec, CPULoadStoreSpec)
            self._bump_group(group_idx, 1, len(output.keys_to_store))
            self._bump_group(group_idx, 2, len(output.evicted_keys))
            block_ids.update(
                zip(output.keys_to_store, map(int, output.store_spec.block_ids))
            )
            evicted_keys.extend(output.evicted_keys)
        stored = [key for key in keys if key in block_ids]
        return PrepareStoreOutput(
            keys_to_store=stored,
            store_spec=CPULoadStoreSpec([block_ids[key] for key in stored]),
            evicted_keys=evicted_keys,
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