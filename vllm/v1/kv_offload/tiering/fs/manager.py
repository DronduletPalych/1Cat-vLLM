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
"""

import functools
import json
import os
import threading
from collections import OrderedDict
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    RequestOffloadingContext,
    SecondaryTierManager,
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
        n_read_threads: int = 16,
        n_write_threads: int = 16,
        max_bytes: int | None = None,
    ):
        """
        Args:
            offloading_spec: contains the vllm_config, kv_cache_config
                and block_size_factor.
            primary_kv_view: Memoryview of the primary tier's CPU KV cache.
            tier_type: Tier type identifier, set by SecondaryTierFactory.
            root_dir: Root directory for block files.
            n_read_threads: Number of read-priority I/O threads.
            n_write_threads: Number of write-priority I/O threads.
            max_bytes: Byte budget for stored block data. None keeps the
                historical unbounded behaviour.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)

        # Extract block size from primary view
        assert primary_kv_view.strides is not None, (
            "primary_kv_view.strides cannot be None"
        )
        self._block_size: int = primary_kv_view.strides[0]

        # Create file mapper
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

        if isinstance(max_bytes, bool) or (
            max_bytes is not None and not isinstance(max_bytes, int)
        ):
            raise TypeError("max_bytes must be a non-negative integer or None")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer or None")
        self.max_bytes = max_bytes
        self._capacity_lock = threading.Lock()
        self._entries: OrderedDict[str, int] = OrderedDict()
        self._cache_bytes = 0
        self._reserved_bytes = 0
        self._protected: dict[str, int] = {}
        self._writing: set[str] = set()
        if max_bytes is not None:
            self._scan_cache()

        self._pool = DualQueueThreadPool(
            n_read_threads,
            n_write_threads,
            thread_name_prefix="vllm_kv_py_fs",
        )

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(
        self, key: OffloadKey, req_context: ReqContext | None = None
    ) -> bool | None:
        return os.path.exists(self.file_mapper.get_file_name(key))

    def _block_roots(self) -> list[str]:
        """Directories that actually hold block files.

        `config.json` sits in `<base>/`, but blocks are written to
        `<base>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin` (file_mapper.py:108), so
        scanning `dirname(config_path)` finds nothing and the capacity cap never
        applies. The first night with the tier on showed exactly that: the
        counter reported 0 blocks while the directory grew to 42 GiB, filled the
        filesystem to 100%, and every store after that failed with
        "Short write: expected 27262976 bytes, wrote 2281472".

        The block root is derived from a real key rather than guessed, so it
        follows the mapper if the layout changes.
        """
        base = os.path.dirname(self.file_mapper.get_config_file_path())
        rank = getattr(self.file_mapper, "rank", 0)
        roots = [f"{base}_r{rank}"]
        if os.path.isdir(base):
            for name in os.listdir(base):
                candidate = os.path.join(base, name)
                if name.startswith(os.path.basename(base) + "_r") and os.path.isdir(
                    candidate
                ):
                    roots.append(candidate)
        return sorted(set(roots))

    def _scan_cache(self) -> None:
        """Build the LRU index once at startup, oldest first.

        Recency comes from modification time so a restart does not evict what
        it has just written. Files that cannot be stat'ed are skipped rather
        than failing startup.
        """
        entries = []
        for root in self._block_roots():
            for dirpath, _, filenames in os.walk(root):
                for name in filenames:
                    if not name.endswith(".bin"):
                        continue
                    entry = os.path.join(dirpath, name)
                    try:
                        st = os.stat(entry)
                    except OSError:
                        continue
                    entries.append((st.st_mtime_ns, entry, st.st_size))
        entries.sort()
        for _, entry, size in entries:
            self._entries[entry] = size
        self._cache_bytes = sum(self._entries.values())
        logger.info(
            "Filesystem KV tier accounting: %d blocks, %.2f GiB, max_bytes=%s, "
            "roots=%s",
            len(self._entries),
            self._cache_bytes / (1 << 30),
            self.max_bytes,
            self._block_roots(),
        )

    def _evict_locked(self, need: int, protected: Collection[str]) -> int:
        """Remove least-recently-used files until `need` bytes are free.

        The caller holds _capacity_lock. Returns the bytes actually freed.
        Paths that are protected or being written are skipped, so a block in
        flight is never removed underneath a reader or writer.
        """
        if need <= 0:
            return 0
        freed = 0
        for entry in list(self._entries):
            if freed >= need:
                break
            if entry in protected or entry in self._protected:
                continue
            if entry in self._writing:
                continue
            size = self._entries.get(entry, 0)
            try:
                os.remove(entry)
            except FileNotFoundError:
                pass
            except OSError:
                continue
            self._entries.pop(entry, None)
            self._cache_bytes -= size
            freed += size
        return freed

    def _protect(self, paths: Iterable[str]) -> None:
        for entry in paths:
            self._protected[entry] = self._protected.get(entry, 0) + 1

    def _unprotect(self, paths: Iterable[str]) -> None:
        for entry in paths:
            count = self._protected.get(entry, 0) - 1
            if count > 0:
                self._protected[entry] = count
            else:
                self._protected.pop(entry, None)

    def _store_batch(self, job_metadata: JobMetadata) -> None:
        """Store one batch, reserving capacity and evicting LRU when bounded.

        With max_bytes unset this is the old path: no accounting, no eviction.
        A batch that does not fit is skipped with a warning; the request falls
        back to recomputation, and the tier stays consistent.
        """
        paths = [self.file_mapper.get_file_name(k) for k in job_metadata.keys]
        offsets = [int(bid) * self._block_size for bid in job_metadata.block_ids]

        if self.max_bytes is None:
            for entry, offset in zip(paths, offsets):
                store_block(entry, self._primary_kv_view, offset, self._block_size)
            return

        fresh = []
        with self._capacity_lock:
            if self._writing.intersection(paths):
                logger.warning(
                    "Filesystem KV store skipped for job %s: overlapping write.",
                    job_metadata.job_id,
                )
                return
            for entry in paths:
                known = self._entries.pop(entry, None)
                if known is not None:
                    if os.path.exists(entry):
                        self._entries[entry] = known
                        continue
                    self._cache_bytes -= known
                if os.path.exists(entry):
                    try:
                        size = os.path.getsize(entry)
                    except OSError:
                        fresh.append(entry)
                    else:
                        self._cache_bytes += size
                        self._entries[entry] = size
                else:
                    fresh.append(entry)

            needed = len(fresh) * self._block_size
            if needed == 0:
                return
            if needed > self.max_bytes:
                logger.warning(
                    "Filesystem KV store skipped for job %s: batch of %d bytes "
                    "exceeds max_bytes=%d.",
                    job_metadata.job_id,
                    needed,
                    self.max_bytes,
                )
                return
            required = max(
                self._cache_bytes + self._reserved_bytes + needed - self.max_bytes, 0
            )
            freed = self._evict_locked(required, paths)
            if freed < required:
                logger.warning(
                    "Filesystem KV store skipped for job %s: could not free %d "
                    "bytes (freed %d), max_bytes=%d.",
                    job_metadata.job_id,
                    required,
                    freed,
                    self.max_bytes,
                )
                return
            self._reserved_bytes += needed
            self._protect(fresh)
            self._writing.update(fresh)

        try:
            for entry, offset in zip(fresh, offsets):
                store_block(entry, self._primary_kv_view, offset, self._block_size)
        finally:
            with self._capacity_lock:
                self._reserved_bytes -= needed
                self._unprotect(fresh)
                self._writing.difference_update(fresh)
                for entry in fresh:
                    try:
                        size = os.path.getsize(entry)
                    except OSError:
                        size = 0
                    if size:
                        self._cache_bytes += size
                        self._entries[entry] = size

    def submit_store(self, job_metadata: JobMetadata) -> None:
        task = functools.partial(self._store_batch, job_metadata)
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    def submit_load(self, job_metadata: JobMetadata) -> None:
        if self.max_bytes is not None:
            self._protect(self.file_mapper.get_file_name(k) for k in job_metadata.keys)
        tasks = (
            functools.partial(
                load_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids)
        )
        self._pool.enqueue_load(job_metadata.job_id, len(job_metadata.keys), tasks)

    def get_finished_jobs(self) -> Iterable[JobResult]:
        """
        Collect completed jobs from the finished-jobs queue.
        """
        results = [
            JobResult(job_id=job_id, success=success)
            for job_id, success in self._pool.get_finished()
        ]
        if self._protected:
            with self._capacity_lock:
                self._protected.clear()
        return results

    def shutdown(self) -> None:
        """
        Release resources held by this tier.

        Shuts down the thread pool, clearing pending tasks and waiting for
        active threads to complete.
        """
        self._pool.shutdown(wait=True)
