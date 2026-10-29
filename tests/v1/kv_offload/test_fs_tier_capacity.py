# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the filesystem KV tier capacity accounting.

These tests exercise :class:`DirCapacity` directly: reservation, LRU/FIFO
eviction, the soft-overshoot window and the startup scan that rebuilds usage
from the directory contents. They touch a real temporary directory but no KV
tensors, so they run without a GPU.
"""

import os

import pytest

from vllm.v1.kv_offload.tiering.fs.capacity import (
    FS_BLOCK,
    GIB,
    DirCapacity,
    get_dir_capacity,
)

_KIB = 1024
_FILE_BYTES = 4 * _KIB  # size of one block file in these tests
_ENTRY = FS_BLOCK * 2  # rounded file + inode block


@pytest.fixture
def capacity(tmp_path):
    # Four block files fit exactly; one more may overshoot by a single slot.
    cap = DirCapacity(str(tmp_path), max_bytes=4 * _ENTRY, max_overshoot_bytes=_ENTRY)
    yield cap
    cap.shutdown()


def _path(root, name):
    return os.path.join(str(root), name)


def _write(path, size=_FILE_BYTES, mtime=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"x" * size)
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _commit(cap, root, name):
    """Reserve and commit one block file, as a successful store would."""
    path = _path(root, name)
    assert cap.try_reserve(path, _ENTRY)
    _write(path)
    cap.commit_store(path, _ENTRY)
    return path


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def test_entry_size_rounds_to_block_and_adds_inode():
    assert DirCapacity.entry_size_of(_FILE_BYTES) == _ENTRY
    assert DirCapacity.entry_size_of(FS_BLOCK) == _ENTRY
    assert DirCapacity.entry_size_of(FS_BLOCK + 1) == FS_BLOCK * 3


def test_reserve_commit_tracks_usage(capacity, tmp_path):
    path = _commit(capacity, tmp_path, "a")
    snap = capacity.snapshot()
    assert snap["fs_bytes_used"] == _ENTRY
    assert snap["fs_files"] == 1
    assert os.path.exists(path)


def test_failed_store_releases_reserved_bytes(capacity, tmp_path):
    path = _path(tmp_path, "a")
    assert capacity.try_reserve(path, _ENTRY)
    capacity.fail_store(path)
    snap = capacity.snapshot()
    assert snap["fs_bytes_used"] == 0
    assert snap["fs_store_failed_total"] == 1
    assert snap["fs_files"] == 0


def test_abandoned_reservation_releases_reserved_bytes(capacity, tmp_path):
    path = _path(tmp_path, "a")
    assert capacity.try_reserve(path, _ENTRY)
    capacity.abandon_reservation(path)
    assert capacity.snapshot()["fs_bytes_used"] == 0


def test_committed_block_is_not_reserved_twice(capacity, tmp_path):
    _commit(capacity, tmp_path, "a")
    assert capacity.try_reserve(_path(tmp_path, "a"), _ENTRY)
    assert capacity.snapshot()["fs_bytes_used"] == _ENTRY


# ---------------------------------------------------------------------------
# Eviction policy
# ---------------------------------------------------------------------------


def test_store_evicts_oldest_never_read_block(capacity, tmp_path):
    paths = [_commit(capacity, tmp_path, name) for name in ("a", "b", "c", "d")]
    assert capacity.snapshot()["fs_bytes_used"] == 4 * _ENTRY

    _commit(capacity, tmp_path, "e")

    assert not os.path.exists(paths[0])
    assert all(os.path.exists(p) for p in paths[1:])
    snap = capacity.snapshot()
    assert snap["fs_evictions_total"] == 1
    assert snap["fs_bytes_used"] == 4 * _ENTRY


def test_read_blocks_are_evicted_after_never_read_ones(capacity, tmp_path):
    paths = {}
    # "a" and "b" were read; "c" and "d" were only written.
    for name in ("a", "b", "c", "d"):
        paths[name] = _commit(capacity, tmp_path, name)
    capacity.touch(paths["a"])
    capacity.touch(paths["b"])

    evicted = []
    for name in ("e", "f"):
        _commit(capacity, tmp_path, name)
        evicted.append([n for n, p in paths.items() if not os.path.exists(p)])

    # Never-read blocks go first, oldest first; the read blocks are still there
    # even though they were written before "c" and "d".
    assert evicted[0] == ["c"]
    assert evicted[1] == ["c", "d"]
    assert os.path.exists(paths["a"])
    assert os.path.exists(paths["b"])


def test_touch_refreshes_recency_of_read_block(capacity, tmp_path):
    paths = {}
    for name in ("a", "b", "c", "d"):
        paths[name] = _commit(capacity, tmp_path, name)
        capacity.touch(paths[name])

    # The cache is full of read blocks: the least recently read one goes first.
    paths["e"] = _commit(capacity, tmp_path, "e")
    capacity.touch(paths["e"])
    assert not os.path.exists(paths["a"])

    # Re-touching "d" makes "b" the least recently read block.
    capacity.touch(paths["d"])
    _commit(capacity, tmp_path, "f")
    assert not os.path.exists(paths["b"])
    assert os.path.exists(paths["c"])
    assert os.path.exists(paths["d"])


# ---------------------------------------------------------------------------
# Soft overshoot and rejection
# ---------------------------------------------------------------------------


def test_overshoot_is_allowed_for_one_slot(capacity, tmp_path):
    # Nothing is committed, so there is no eviction candidate: the store may
    # overshoot by the configured window instead of failing.
    for name in ("a", "b", "c", "d", "e"):
        assert capacity.try_reserve(_path(tmp_path, name), _ENTRY)

    snap = capacity.snapshot()
    assert snap["fs_bytes_used"] == 5 * _ENTRY
    assert snap["fs_overshoot_bytes"] == _ENTRY
    assert snap["fs_peak_overshoot_bytes"] == _ENTRY
    assert snap["fs_store_rejected_total"] == 0


def test_store_is_rejected_when_overshoot_is_exhausted(capacity, tmp_path):
    for name in ("a", "b", "c", "d", "e"):
        assert capacity.try_reserve(_path(tmp_path, name), _ENTRY)

    assert not capacity.try_reserve(_path(tmp_path, "f"), _ENTRY)
    assert capacity.snapshot()["fs_store_rejected_total"] == 1


def test_trim_returns_to_limit_after_jobs_finish(capacity, tmp_path):
    paths = []
    for name in ("a", "b", "c", "d"):
        paths.append(_commit(capacity, tmp_path, name))
    assert capacity.try_reserve(_path(tmp_path, "e"), _ENTRY)

    capacity.trim()

    assert capacity.snapshot()["fs_bytes_used"] == 4 * _ENTRY
    assert not os.path.exists(paths[0])


# ---------------------------------------------------------------------------
# Startup scan and process-wide sharing
# ---------------------------------------------------------------------------


def test_scan_rebuilds_usage_and_trims_to_limit(tmp_path):
    for i in range(6):
        _write(_path(tmp_path, f"f{i}"), mtime=1000 + i)

    cap = DirCapacity(str(tmp_path), max_bytes=4 * _ENTRY)
    try:
        snap = cap.snapshot()
        assert snap["fs_files"] == 4
        assert snap["fs_bytes_used"] == 4 * _ENTRY
        assert snap["fs_evictions_total"] == 2
        # The two oldest files were dropped, the newest four kept.
        assert not os.path.exists(_path(tmp_path, "f0"))
        assert not os.path.exists(_path(tmp_path, "f1"))
        assert os.path.exists(_path(tmp_path, "f5"))
    finally:
        cap.shutdown()


def test_startup_scan_ignores_tmp_files_and_lock(tmp_path):
    _write(_path(tmp_path, "a"))
    _write(_path(tmp_path, "b.tmp"))

    cap = DirCapacity(str(tmp_path), max_bytes=4 * _ENTRY)
    try:
        snap = cap.snapshot()
        assert snap["fs_files"] == 1
        assert snap["fs_bytes_used"] == _ENTRY
    finally:
        cap.shutdown()


def test_get_dir_capacity_shares_one_object_per_directory(tmp_path):
    first = get_dir_capacity(str(tmp_path), 4 * _ENTRY)
    second = get_dir_capacity(str(tmp_path), 4 * _ENTRY)
    try:
        assert first is second
        assert first.snapshot()["fs_instances"] == 2
    finally:
        first.shutdown()


def test_get_dir_capacity_rejects_conflicting_limit(tmp_path):
    cap = get_dir_capacity(str(tmp_path), 4 * _ENTRY)
    try:
        with pytest.raises(ValueError):
            get_dir_capacity(str(tmp_path), 8 * _ENTRY)
    finally:
        cap.shutdown()


def test_get_dir_capacity_requires_max_bytes(tmp_path):
    with pytest.raises(ValueError):
        get_dir_capacity(str(tmp_path), None)


def test_second_capacity_object_on_same_directory_is_refused(tmp_path):
    cap = DirCapacity(str(tmp_path), max_bytes=4 * _ENTRY)
    try:
        with pytest.raises(RuntimeError):
            DirCapacity(str(tmp_path), max_bytes=4 * _ENTRY)
    finally:
        cap.shutdown()


def test_default_overshoot_is_one_gib(tmp_path):
    cap = DirCapacity(str(tmp_path), max_bytes=64 * GIB)
    try:
        assert cap.max_overshoot_bytes == GIB
    finally:
        cap.shutdown()
