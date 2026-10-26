# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission watermark: the per-request output reservation is capped."""

from types import SimpleNamespace
from typing import cast

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import Request


def _manager(cap: int) -> KVCacheManager:
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.watermark_output_reserve = cap
    return manager


def _req(max_tokens: int | None = None) -> Request:
    attrs = {} if max_tokens is None else {"max_tokens": max_tokens}
    return cast(Request, SimpleNamespace(**attrs))


def test_caps_implicit_full_context_request():
    """A request without max_tokens asks for the rest of the window (168340
    here); the cap keeps one request from reserving the whole pool."""
    assert _manager(32768)._reserved_output_tokens(_req(168340)) == 32768


def test_keeps_smaller_explicit_request():
    assert _manager(32768)._reserved_output_tokens(_req(512)) == 512


def test_zero_restores_uncapped_reservation():
    assert _manager(0)._reserved_output_tokens(_req(168340)) == 168340


def test_missing_max_tokens_reserves_nothing_extra():
    assert _manager(32768)._reserved_output_tokens(_req()) == 0
