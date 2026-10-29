# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
FileSystemTierManager: Pure-Python file system secondary tier for KV cache offloading.

Store path:
    Data is written to a temp file (<dest_path.tmp>) via os.write,
    then os.replace'd to the final path (without .tmp).

Load path:
    Data is read from the block file directly via os.readv into the
    provided memoryview slice.

File naming:  <base_path>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash_hex>.bin
              (hash-based subdirectories to limit directory fan-out)

Capacity: every instance in the scheduler process shares one DirCapacity
object per realpath(root_dir) (see capacity.py). Space is reserved before a
write is queued, released if the write fails, and the cache is trimmed back to
``max_bytes`` after every finished job.
"""

import functools
import json
import os
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    get_offload_block_hash,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    SecondaryTierManager,
)
from vllm.v1.kv_offload.tiering.fs.capacity import (
    DEFAULT_STATE_MAX_BYTES,
    GIB,
    DirCapacity,
    get_dir_capacity,
)
from vllm.v1.kv_offload.tiering.fs.io import load_block, store_block
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec

logger = init_logger(__name__)


class FileSystemTierManager(SecondaryTierManager):
    """
    Pure-Python disk-backed secondary tier.

    Read-priority threads service load jobs preferentially; write-priority
    threads service store jobs preferentially.  Both groups can drain either
    queue, so neither starves.

    submit_store / submit_load are non-blocking: they enqueue tasks and return.
    get_finished_jobs() polls job completion and returns completed JobResults.

    """

    def __init__(
        self,
        offloading_spec: "OffloadingSpec",
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        max_bytes: int | None = None,
        max_overshoot_bytes: int = GIB,
        state_max_bytes: int = DEFAULT_STATE_MAX_BYTES,
        n_read_threads: int = 16,
        n_write_threads: int = 16,
    ):
        """
        Args:
            offloading_spec: contains the vllm_config, kv_cache_config
                and block_size_factor.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            max_bytes: Hard capacity of the whole cache directory (required).
            max_overshoot_bytes: How far one job may push the cache above
                max_bytes before stores are rejected.
            state_max_bytes: Hard cap on Mamba checkpoint (recurrent state)
                bytes inside the same directory. Checkpoints beyond it are
                dropped oldest-first, together with the attention of the same
                block; 0 disables the cap.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]
        # On-disk cost of one block file (size rounded to 4 KiB + metadata).
        self._slot_bytes: int = DirCapacity.entry_size_of(self._block_size)
        self.root_dir = root_dir

        # Create file mapper
        if max_bytes is None:
            raise ValueError(
                "fs tier requires 'max_bytes' in its configuration: "
                "unbounded FS cache is not allowed"
            )
        if int(max_bytes) <= 0:
            raise ValueError(f"fs tier: max_bytes must be positive, got {max_bytes}")
        self.max_bytes = int(max_bytes)
        if int(state_max_bytes) < 0:
            raise ValueError(
                f"fs tier: state_max_bytes must be >= 0, got {state_max_bytes}"
            )
        self.state_max_bytes = int(state_max_bytes)

        self.file_mapper = FileMapper.from_offloading_spec(
            root_dir=root_dir,
            offloading_spec=offloading_spec,
            gpu_blocks_per_file=offloading_spec.block_size_factor,
        )

        # Write config file
        config_path = self.file_mapper.get_config_file_path()
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(
                    self.file_mapper.get_run_config(), f, indent=2, sort_keys=True
                )

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

        # One shared capacity object (and one directory lock) per realpath.
        group_indices = sorted(
            getattr(offloading_spec, "cpu_group_num_blocks", {}) or {}
        )
        # Mamba groups hold recurrent state, not attention KV: they are
        # accounted apart and capped by state_max_bytes.
        mamba_groups: set[int] = set(
            getattr(offloading_spec, "mamba_groups", ()) or ()
        )
        self._cap = get_dir_capacity(
            root_dir,
            self.max_bytes,
            int(max_overshoot_bytes),
            group_indices=group_indices,
            state_max_bytes=self.state_max_bytes,
            mamba_groups=mamba_groups,
        )
        # Paths whose store job was queued but has not finished yet.
        self._pending_paths: set[str] = set()

        # Optional I/O counters (env-gated) so operators can prove that the
        # disk tier is actually storing and restoring blocks.
        self._stats_enabled = os.environ.get("VLLM_FS_TIER_STATS") == "1"
        self._stats_stored_blocks = 0
        self._stats_loaded_blocks = 0
        self._stats_lookups = 0
        self._stats_lookup_hits = 0
        self._stats_last_log = time.monotonic()

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def _path(self, key: OffloadKey) -> str:
        return self.file_mapper.get_file_name(key)

    def lookup(
        self, key: OffloadKey, req_context: ReqContext | None = None
    ) -> bool | None:
        name = self._path(key)
        exists = os.path.exists(name)
        if exists:
            # Recency: a looked-up (and therefore soon to be promoted) block is
            # the last candidate for eviction.
            self._cap.touch(name)
        self._cap.note_lookup(exists)
        if self._stats_enabled:
            self._stats_lookups += 1
            if exists:
                self._stats_lookup_hits += 1
            elif self._stats_lookups <= 8 or self._stats_lookups % 500 == 0:
                hex_hash = get_offload_block_hash(key).hex()
                parent = os.path.dirname(os.path.dirname(name))
                elsewhere: list[str] = []
                try:
                    for sub in os.listdir(parent):
                        if os.path.exists(os.path.join(parent, sub, hex_hash + ".bin")):
                            elsewhere.append(sub)
                except OSError:
                    pass
                logger.info(
                    "FS-LOOK-MISS n=%d group=%s hash=%s path=%s elsewhere=%s",
                    self._stats_lookups,
                    get_offload_group_idx(key),
                    hex_hash[:16],
                    name,
                    elsewhere,
                )
        return exists

    def touch(self, keys, req_context=None) -> None:
        """Mark blocks as recently read (LRU recency)."""
        for key in keys:
            self._cap.touch(self._path(key))

    # ---------------------------------------------------------------- stores

    def _store_one(self, path: str, block_id: int) -> None:
        try:
            store_block(path, self._primary_kv_view, block_id, self._block_size)
        except BaseException:
            self._cap.fail_store(path)
            raise
        finally:
            self._pending_paths.discard(path)
        self._cap.commit_store(path, self._slot_bytes)
        self._cap.note_store_block()

    def submit_store(self, job_metadata: JobMetadata) -> None:
        """Queue a primary->disk cascade for one job.

        Space is reserved for every block before anything is queued. If the
        cache cannot make room even with the allowed overshoot, the whole job
        is reported as failed and no block of it is marked stored.
        """
        reserved: list[tuple[str, int]] = []
        tasks = []
        for key, bid in zip(job_metadata.keys, job_metadata.block_ids):
            path = self._path(key)
            block_id = int(bid) * self._block_size
            if path in self._pending_paths:
                continue
            if os.path.exists(path):
                # Already on disk (and already accounted by the startup scan or
                # by an earlier commit): nothing to write, nothing to reserve.
                self._cap.touch(path)
                continue
            if not self._cap.try_reserve(path, self._slot_bytes):
                for res_path, _ in reserved:
                    self._cap.abandon_reservation(res_path)
                for res_path, _ in reserved:
                    self._pending_paths.discard(res_path)
                logger.warning(
                    "FS-STORE-REJECT job=%s keys=%d used=%d limit=%d",
                    job_metadata.job_id,
                    len(job_metadata.keys),
                    self._cap.snapshot()["fs_bytes_used"],
                    self.max_bytes,
                )
                # Report the job as failed so the framework unpins the blocks
                # and never treats them as stored.
                self._pool.fail_job(job_metadata.job_id)
                return
            reserved.append((path, self._slot_bytes))
            self._pending_paths.add(path)
            tasks.append(functools.partial(self._store_one, path, block_id))

        if not tasks:
            # Everything was already present: a no-op job still has to report
            # completion, otherwise the framework keeps the blocks pinned.
            self._pool.finish_empty_job(job_metadata.job_id)
            return

        self._pool.enqueue_store(job_metadata.job_id, len(tasks), tasks)

    # ----------------------------------------------------------------- loads

    def _load_one(self, path: str, block_id: int) -> None:
        self._cap.acquire_read(path)
        try:
            load_block(path, self._primary_kv_view, block_id, self._block_size)
        except BaseException:
            self._cap.note_load_error()
            self._cap.drop_entry(path)
            raise
        else:
            self._cap.note_load_block()
        finally:
            self._cap.release_read(path)

    def submit_load(self, job_metadata: JobMetadata) -> None:
        tasks = []
        for key, bid in zip(job_metadata.keys, job_metadata.block_ids):
            path = self._path(key)
            self._cap.touch(path)
            tasks.append(
                functools.partial(self._load_one, path, int(bid) * self._block_size)
            )
        if not tasks:
            self._pool.finish_empty_job(job_metadata.job_id)
            return
        self._pool.enqueue_load(job_metadata.job_id, len(tasks), tasks)

    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        finished = list(self._pool.get_finished())
        # Mandatory trim after every finished job: the cache must fall back to
        # max_bytes as soon as nothing is in flight.
        if finished:
            self._cap.trim()
        return (
            JobResult(job_id=job_id, success=success)
            for job_id, success in finished
        )

    def stats(self) -> dict[str, int]:
        """Aggregate metrics for this directory (shared across instances)."""
        return self._cap.snapshot()

    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the thread pool, clearing pending tasks and waiting for
        active threads to complete.
        """
        self._pool.shutdown(wait=True)
