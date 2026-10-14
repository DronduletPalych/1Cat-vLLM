# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocator snapshot instrumentation for OOM diagnosis.

Enabled via env var ``VLLM_ALLOC_SNAPSHOT_DIR`` (path to output directory).
When not set, all public functions are no-ops with zero overhead.

Captures:
- ``torch.accelerator.memory_stats()`` — cheap O(1) stats
- ``torch.cuda.memory._snapshot()`` — full allocation snapshot (expensive,
  used only on OOM / calibration)
- Per-stream / segment breakdown for free-list analysis

Usage (service env):
    VLLM_ALLOC_SNAPSHOT_DIR=/var/log/vllm/alloc-snapshots

Output files:
    rank{N}_{context}_{timestamp}.json
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# One-shot calibration state per rank (module-level, process-scoped)
_calibration_done: set[int] = set()


def _env_dir() -> str | None:
    raw = os.environ.get("VLLM_ALLOC_SNAPSHOT_DIR", "").strip()
    return raw if raw else None


def is_enabled() -> bool:
    return _env_dir() is not None


def memory_summary(device: torch.device) -> dict[str, Any]:
    """Cheap O(1) memory stats — safe to call on every step."""
    try:
        stats = torch.accelerator.memory_stats(device)
        return {
            "allocated_bytes": stats.get("allocated_bytes.all.current", 0),
            "reserved_bytes": stats.get("reserved_bytes.all.current", 0),
            "active_bytes": stats.get("active_bytes.all.current", 0),
            "inactive_bytes": stats.get("inactive_bytes.all.current", 0),
            "inactive_split_bytes": stats.get("inactive_split_bytes.all.current", 0),
            "segment_count": stats.get("segment_count.all", 0),
            "segment_bytes": stats.get("segment_bytes.all", 0),
            "max_allocated_bytes": stats.get("allocated_bytes.all.peak", 0),
            "max_reserved_bytes": stats.get("reserved_bytes.all.peak", 0),
        }
    except Exception as e:
        return {"error": str(e)}


def capture_snapshot(
    device: torch.device,
    rank: int,
    context: str,
    include_history: bool = False,
) -> dict[str, Any]:
    """Capture full allocator snapshot.

    Args:
        device: CUDA device for this rank.
        rank: Tensor-parallel rank number.
        context: Human-readable tag (e.g. ``"oom_execute_model"``).
        include_history: If True, enable ``_record_memory_history`` before
            snapshot (adds per-allocation overhead; only for OOM / rare events).

    Returns the captured dict (empty if instrumentation disabled).
    """
    out_dir = _env_dir()
    if out_dir is None:
        return {}

    ts = time.time()
    path = Path(out_dir) / f"rank{rank}_{context}_{ts:.6f}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "rank": rank,
        "context": context,
        "timestamp": ts,
        "device": str(device),
        "memory_stats": memory_summary(device),
    }

    # Full PyTorch allocation snapshot
    try:
        if include_history:
            torch.cuda.memory._record_memory_history(max_entries=100_000)
        snap = torch.cuda.memory._snapshot()
        if snap:
            traces = snap.get("device_traces", [])
            if traces:
                result["allocation_events"] = traces[0].get("events", [])
                result["segment_summaries"] = traces[0].get("segments", [])
        if include_history:
            torch.cuda.memory._record_memory_history(enabled=False)
    except Exception as e:
        result["snapshot_error"] = str(e)
        # Ensure history is disabled even on failure
        try:
            torch.cuda.memory._record_memory_history(enabled=False)
        except Exception:
            pass

    # Per-pool / per-stream stats (if available)
    try:
        stats = torch.accelerator.memory_stats(device)
        pool_keys = {k: v for k, v in stats.items() if "pool" in k.lower()}
        if pool_keys:
            result["pool_stats"] = pool_keys
    except Exception:
        pass

    path.write_text(json.dumps(result, indent=2, default=str))
    n_events = len(result.get("allocation_events", []))
    logger.info(
        "AllocSnapshot rank=%d context=%s events=%d -> %s",
        rank, context, n_events, path,
    )
    return result


def on_oom(
    device: torch.device,
    rank: int,
    context: str,
    exc: BaseException,
) -> None:
    """Called when an OOM is caught — guaranteed snapshot with full history.

    This is the critical capture: it records the allocator state at the exact
    moment of failure, splitting ``reserved-unalloc`` into mapped-free /
    deferred-free / fragmented-small blocks.
    """
    if not is_enabled():
        return
    logger.warning(
        "AllocSnapshot: OOM at rank=%d context=%s: %s",
        rank, context, str(exc)[:300],
    )
    try:
        capture_snapshot(device, rank, f"oom_{context}", include_history=True)
    except Exception as e:
        logger.error("AllocSnapshot: failed to capture on OOM: %s", e)


def calibration_capture(device: torch.device, rank: int) -> None:
    """One-shot snapshot on first forward pass per rank.

    Captures the static baseline (weights + KV pool + CUDA graphs +
    non-torch overhead) before any transient activations accumulate.
    """
    if not is_enabled() or rank in _calibration_done:
        return
    _calibration_done.add(rank)
    try:
        capture_snapshot(device, rank, "calibration_first_forward")
    except Exception as e:
        logger.error("AllocSnapshot: calibration capture failed: %s", e)
