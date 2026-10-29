# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process-wide capacity accounting for one FS-tier cache directory.

Design (TZ ред.5, variant A):
  * one counter per realpath(root_dir): all KV-group FS tier instances in the
    scheduler process share a single ``DirCapacity`` object;
  * an exclusive ``flock(LOCK_EX)`` on ``<root_dir>/.fs_tier.lock`` guards the
    directory against a second process;
  * ``used`` is the *real* occupancy of every block file:
    ``round(st_size, 4096) + 4096``;
  * space is reserved before a write is queued and released if the write fails;
  * LRU eviction, blocks that are in flight (being read or written) are never
    evicted; blocks that were written and never read are evicted first;
  * soft overshoot: a job may push ``used`` above ``max_bytes`` but never above
    ``max_bytes + max_overshoot_bytes``; beyond that the job is rejected;
  * after every finished job the cache is trimmed back to ``max_bytes``;
  * a background thread compares the on-disk sum with the committed counter and
    publishes ``fs_accounting_drift_bytes``.

All filesystem mutations happen while holding ``self._lock`` so the walk-based
drift check and the on-line accounting can never interleave destructively.
"""

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)

GIB = 1024**3
# ext4 block: the on-disk cost of a file is its size rounded up to the block
# size plus one inode (4 KiB is the smallest ext4 block we ever see).
FS_BLOCK = 4096
LOCK_NAME = ".fs_tier.lock"

_DRIFT_INTERVAL_S = 60.0
_METRICS_INTERVAL_S = 10.0


@dataclass
class _Entry:
    size: int
    reads: int = 0


class DirCapacity:
    """Capacity reservation, LRU eviction and metrics for one cache directory."""

    def __init__(
        self,
        root_dir: str,
        max_bytes: int,
        max_overshoot_bytes: int = GIB,
        group_indices: list[int] | None = None,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("fs tier: max_bytes must be positive")
        if max_overshoot_bytes < 0:
            raise ValueError("fs tier: max_overshoot_bytes must be >= 0")

        self.root_dir = root_dir
        self.real_root = os.path.realpath(root_dir)
        self.max_bytes = int(max_bytes)
        self.max_overshoot_bytes = int(max_overshoot_bytes)
        # KV group indices of the layout. A block is one file per group, so
        # eviction must remove every group file of a block: a half-block
        # (attention without state) is unreadable and only wastes capacity.
        self._group_indices: tuple[int, ...] = (
            tuple(sorted(group_indices)) if group_indices else ()
        )

        self._lock = threading.RLock()
        # Sizes plus two eviction classes (TZ §3.1): blocks written and never
        # read are evicted first, oldest first (FIFO); blocks that were read
        # are evicted last, least recently read first (LRU).
        self._entries: dict[str, _Entry] = {}
        self._unread: OrderedDict[str, None] = OrderedDict()
        self._read: OrderedDict[str, None] = OrderedDict()
        self._inflight: dict[str, int] = {}
        self._used = 0  # committed + reserved
        self._committed = 0  # committed only (matches on-disk sum)
        self._reserved_sizes: dict[str, int] = {}

        self.instances = 0
        self.evictions = 0
        self.rejected = 0
        self.store_failed = 0
        self.load_errors = 0
        self.overshoot_bytes = 0
        self.drift_bytes = 0
        self.files = 0
        self.stored_blocks = 0
        self.loaded_blocks = 0
        self.lookups = 0
        self.lookup_hits = 0
        self._last_drift_scan = 0.0
        self._last_metrics_log = time.monotonic()

        os.makedirs(self.real_root, exist_ok=True)
        self._lock_fd = self._acquire_dir_lock()
        self._scan()
        self._trim_locked()

        logger.info(
            "FS-CAP pid=%d root_dir=%s used=%d limit=%d overshoot_limit=%d "
            "files=%d lock=acquired",
            os.getpid(),
            self.real_root,
            self._used,
            self.max_bytes,
            self.max_overshoot_bytes,
            self.files,
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="vllm_fs_tier_capacity",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------ setup

    def _acquire_dir_lock(self) -> int:
        import fcntl

        path = os.path.join(self.real_root, LOCK_NAME)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise RuntimeError(
                f"FS-tier cache directory {self.real_root} is already used by "
                f"another process (lock file {path}): {exc}"
            ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        return fd

    @staticmethod
    def _entry_size(st_size: int) -> int:
        return ((st_size + FS_BLOCK - 1) // FS_BLOCK) * FS_BLOCK + FS_BLOCK

    @staticmethod
    def entry_size_of(size: int) -> int:
        """On-disk cost of a file of *size* bytes (rounded + inode block)."""
        return DirCapacity._entry_size(size)

    def trim(self) -> None:
        """Evict LRU blocks until ``used <= max_bytes``."""
        with self._lock:
            self._trim_locked()

    def _scan(self) -> None:
        """Rebuild ``used`` from the real directory contents."""
        total = 0
        files = 0
        found: list[tuple[float, str, int]] = []
        for dirpath, _dirnames, filenames in os.walk(self.real_root):
            for name in filenames:
                if name.endswith(".tmp") or name == LOCK_NAME:
                    continue
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                size = self._entry_size(st.st_size)
                total += size
                files += 1
                found.append((st.st_mtime, path, size))
        self._entries.clear()
        self._unread.clear()
        self._read.clear()
        # atime is useless under noatime; restore the order from mtime.
        found.sort()
        for _mtime, path, size in found:
            self._entries[path] = _Entry(size=size, reads=0)
            self._unread[path] = None
        self._used = total
        self._committed = total
        self.files = files
        self._last_drift_scan = time.monotonic()
        logger.info(
            "FS-CAP-SCAN root_dir=%s used=%d files=%d",
            self.real_root,
            total,
            files,
        )

    def register_instance(self) -> int:
        with self._lock:
            self.instances += 1
            count = self.instances
        logger.info(
            "FS-CAP pid=%d root_dir=%s instances=%d limit=%d lock=acquired",
            os.getpid(),
            self.real_root,
            count,
            self.max_bytes,
        )
        return count

    # ------------------------------------------------------------- eviction

    def set_group_indices(self, group_indices: list[int] | None) -> None:
        """Adopt the layout's group list once a tier manager knows it."""
        with self._lock:
            if not self._group_indices and group_indices:
                self._group_indices = tuple(sorted(group_indices))

    def _bundle_paths_locked(self, path: str) -> list[str]:
        """Every group file of the same block, derived from the naming.

        Layout: <root>/<base>_r<rank>/<hhh>/<hh>_g<group>/<hash>.bin. The
        block hash is the file name, so all group files of one block live in
        sibling directories of the victim's own directory.
        """
        if not self._group_indices:
            return [path]
        group_dir = os.path.dirname(path)
        block_dir = os.path.dirname(group_dir)
        head, sep, tail = os.path.basename(group_dir).rpartition("_g")
        if not sep or not tail.isdigit():
            return [path]
        fname = os.path.basename(path)
        return [
            os.path.join(block_dir, f"{head}_g{group}", fname)
            for group in self._group_indices
        ]

    def _pick_victim_locked(self) -> str | None:
        for queue in (self._unread, self._read):
            for path in queue:
                if path not in self._entries or self._inflight.get(path):
                    continue
                # Never split a block: skip the bundle if any of its files is
                # reserved or being read right now.
                if any(self._inflight.get(p) for p in self._bundle_paths_locked(path)):
                    continue
                return path
        return None

    def _evict_one_locked(self) -> bool:
        path = self._pick_victim_locked()
        if path is None:
            return False
        removed = 0
        for victim in self._bundle_paths_locked(path):
            entry = self._entries.get(victim)
            if entry is None or self._inflight.get(victim):
                continue
            try:
                os.remove(victim)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("FS-EVICT-FAIL path=%s err=%s", victim, exc)
                continue
            self._entries.pop(victim, None)
            self._unread.pop(victim, None)
            self._read.pop(victim, None)
            self._used -= entry.size
            self._committed -= entry.size
            self.files -= 1
            self.evictions += 1
            removed += 1
        return removed > 0

    def _make_room_locked(self, need: int) -> None:  # noqa: D401
        """Evict LRU blocks until ``need`` more bytes fit under the limit."""
        while self._used + need > self.max_bytes:
            if not self._evict_one_locked():
                return

    def _trim_locked(self) -> None:
        """Enforce ``used <= max_bytes`` once no job is waiting on space."""
        while self._used > self.max_bytes:
            if not self._evict_one_locked():
                return

    # ------------------------------------------------------------ reservation

    def try_reserve(self, path: str, size: int) -> bool:
        """Reserve *size* bytes for *path*; False if the hard cap is hit."""
        with self._lock:
            if path in self._entries:
                # Already stored: nothing to do, nothing to reserve.
                return True
            if self._inflight.get(path):
                return True
            self._make_room_locked(size)
            used_after = self._used + size
            if used_after > self.max_bytes + self.max_overshoot_bytes:
                self.rejected += 1
                logger.warning(
                    "FS-CAP-REJECT used=%d limit=%d job_bytes=%d overshoot_limit=%d",
                    self._used,
                    self.max_bytes,
                    size,
                    self.max_overshoot_bytes,
                )
                return False
            self._used = used_after
            self._reserved_sizes[path] = size
            self._inflight[path] = self._inflight.get(path, 0) + 1
            if used_after > self.max_bytes:
                self.overshoot_bytes = max(
                    self.overshoot_bytes, used_after - self.max_bytes
                )
                logger.warning(
                    "FS-CAP-OVERSHOOT used=%d limit=%d job_bytes=%d",
                    used_after,
                    self.max_bytes,
                    size,
                )
            return True

    def _release_inflight_locked(self, path: str) -> None:
        refs = self._inflight.get(path, 0)
        if refs <= 1:
            self._inflight.pop(path, None)
        else:
            self._inflight[path] = refs - 1

    def abandon_reservation(self, path: str) -> None:
        """Drop a reservation whose job will never be written (job rejected)."""
        with self._lock:
            if not self._inflight.get(path):
                return
            self._release_inflight_locked(path)
            if path not in self._entries:
                self._used -= self._reserved_sizes.pop(path, 0)

    def commit_store(self, path: str, size: int) -> None:
        """A store finished successfully: move the reservation into the LRU."""
        with self._lock:
            self._release_inflight_locked(path)
            self._reserved_sizes.pop(path, None)
            entry = self._entries.get(path)
            if entry is None:
                self._entries[path] = _Entry(size=size, reads=0)
                # Newly written, never read: oldest-unread goes first.
                self._unread[path] = None
                self._committed += size
                self.files += 1

    def fail_store(self, path: str) -> None:
        """A store failed: give the reserved bytes back."""
        with self._lock:
            self._release_inflight_locked(path)
            if path in self._entries:
                return
            self._used -= self._reserved_sizes.pop(path, 0)
            self.store_failed += 1

    # --------------------------------------------------------------- reads

    def acquire_read(self, path: str) -> None:
        with self._lock:
            self._inflight[path] = self._inflight.get(path, 0) + 1

    def release_read(self, path: str) -> None:
        with self._lock:
            refs = self._inflight.get(path, 0)
            if refs <= 1:
                self._inflight.pop(path, None)
            else:
                self._inflight[path] = refs - 1

    def touch(self, path: str) -> None:
        """Mark a block as read: it moves from the FIFO to the LRU class."""
        with self._lock:
            entry = self._entries.get(path)
            if entry is None:
                return
            entry.reads += 1
            self._unread.pop(path, None)
            self._read[path] = None
            self._read.move_to_end(path)

    def note_store_block(self, count: int = 1) -> None:
        with self._lock:
            self.stored_blocks += count

    def note_load_block(self, count: int = 1) -> None:
        with self._lock:
            self.loaded_blocks += count

    def note_lookup(self, hit: bool) -> None:
        with self._lock:
            self.lookups += 1
            if hit:
                self.lookup_hits += 1

    def note_load_error(self) -> None:
        with self._lock:
            self.load_errors += 1

    def drop_entry(self, path: str) -> None:
        """Remove a file that failed to load from the accounting."""
        with self._lock:
            entry = self._entries.pop(path, None)
            self._unread.pop(path, None)
            self._read.pop(path, None)
            if entry is None:
                return
            self._used -= entry.size
            self._committed -= entry.size
            self.files -= 1

    # --------------------------------------------------------------- metrics

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "fs_bytes_used": self._used,
                "fs_bytes_limit": self.max_bytes,
                "fs_overshoot_bytes": max(0, self._used - self.max_bytes),
                "fs_evictions_total": self.evictions,
                "fs_store_rejected_total": self.rejected,
                "fs_store_failed_total": self.store_failed,
                "fs_load_errors_total": self.load_errors,
                "fs_accounting_drift_bytes": self.drift_bytes,
                "fs_instances": self.instances,
                "fs_files": self.files,
                "fs_stored_blocks": self.stored_blocks,
                "fs_loaded_blocks": self.loaded_blocks,
                "fs_lookups": self.lookups,
                "fs_lookup_hits": self.lookup_hits,
                "fs_inflight": len(self._inflight),
                "fs_peak_overshoot_bytes": self.overshoot_bytes,
            }

    def log_metrics(self) -> None:
        snap = self.snapshot()
        logger.info(
            "FS-METRICS used=%d limit=%d overshoot=%d peak_overshoot=%d "
            "evictions=%d rejected=%d store_failed=%d load_errors=%d "
            "drift=%d files=%d stored_blocks=%d loaded_blocks=%d "
            "lookups=%d lookup_hits=%d inflight=%d instances=%d",
            snap["fs_bytes_used"],
            snap["fs_bytes_limit"],
            snap["fs_overshoot_bytes"],
            snap["fs_peak_overshoot_bytes"],
            snap["fs_evictions_total"],
            snap["fs_store_rejected_total"],
            snap["fs_store_failed_total"],
            snap["fs_load_errors_total"],
            snap["fs_accounting_drift_bytes"],
            snap["fs_files"],
            snap["fs_stored_blocks"],
            snap["fs_loaded_blocks"],
            snap["fs_lookups"],
            snap["fs_lookup_hits"],
            snap["fs_inflight"],
            snap["fs_instances"],
        )

    def drift_check(self) -> int:
        """Compare the committed counter with the real on-disk sum."""
        on_disk = 0
        files = 0
        for dirpath, _dirnames, filenames in os.walk(self.real_root):
            for name in filenames:
                if name.endswith(".tmp") or name == LOCK_NAME:
                    continue
                try:
                    st = os.stat(os.path.join(dirpath, name))
                except OSError:
                    continue
                on_disk += self._entry_size(st.st_size)
                files += 1
        with self._lock:
            drift = on_disk - self._committed
            self.drift_bytes = drift
            self.files = files
        limit = self.max_bytes
        if abs(drift) > limit // 100:
            logger.error(
                "FS-DRIFT accounting mismatch: on_disk=%d committed=%d drift=%d "
                "(limit=%d, tolerance=%d) root_dir=%s",
                on_disk,
                self._committed,
                drift,
                limit,
                limit // 100,
                self.real_root,
            )
        else:
            logger.info(
                "FS-DRIFT ok on_disk=%d committed=%d drift=%d files=%d",
                on_disk,
                self._committed,
                drift,
                files,
            )
        return drift

    def _monitor_loop(self) -> None:
        while not self._stop.wait(1.0):
            now = time.monotonic()
            if now - self._last_drift_scan >= _DRIFT_INTERVAL_S:
                self._last_drift_scan = now
                try:
                    self.drift_check()
                except Exception as exc:  # pragma: no cover - diagnostics
                    logger.warning("FS-DRIFT-SCAN-FAIL %s", exc)
            if now - self._last_metrics_log >= _METRICS_INTERVAL_S:
                self._last_metrics_log = now
                try:
                    self.log_metrics()
                    with self._lock:
                        self._trim_locked()
                except Exception as exc:  # pragma: no cover - diagnostics
                    logger.warning("FS-METRICS-FAIL %s", exc)

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._lock_fd  # noqa: B018 - keep the lock until process exit
        except AttributeError:
            return


_DIR_STATE: dict[str, DirCapacity] = {}
_DIR_STATE_LOCK = threading.Lock()


def get_dir_capacity(
    root_dir: str,
    max_bytes: int | None,
    max_overshoot_bytes: int = GIB,
    group_indices: list[int] | None = None,
) -> DirCapacity:
    """Return the process-wide capacity object for *root_dir*."""
    real = os.path.realpath(root_dir)
    with _DIR_STATE_LOCK:
        state = _DIR_STATE.get(real)
        if state is None:
            if max_bytes is None:
                raise ValueError(
                    "fs tier requires 'max_bytes' in its configuration: "
                    "unbounded FS cache is not allowed"
                )
            state = DirCapacity(
                real,
                int(max_bytes),
                int(max_overshoot_bytes),
                group_indices=group_indices,
            )
            _DIR_STATE[real] = state
        elif max_bytes is not None and int(max_bytes) != state.max_bytes:
            raise ValueError(
                f"fs tier: conflicting max_bytes for {real}: "
                f"{int(max_bytes)} != {state.max_bytes}"
            )
        state.set_group_indices(group_indices)
        state.register_instance()
        return state
