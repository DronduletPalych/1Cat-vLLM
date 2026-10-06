# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional prompt-ngram assist for the MTP (Eagle) drafter.

MTP analogue of the fork's DFlash2 ``ngram_assist``: alongside the model-based
MTP draft, a cheap prompt-ngram (prompt-lookup) pass finds, for each request,
the longest recent suffix of its token history that has already occurred and
proposes the tokens that followed it.

Policy (conservative, bonus-only):
  * If the lookup yields a *full* run of ``num_speculative_tokens`` tokens
    (a verbatim-history continuation), it REPLACES the MTP draft for that
    request. Such a continuation is at least as reliable as the model draft
    and costs no GPU work.
  * Otherwise the MTP draft is untouched (the common case on prose).

Every proposed token is still verified by the target model, so a bad ngram
guess is simply rejected -- the assist can only add tokens, never corrupt
output.

Enabled by the speculative-config flag (``ngram_assist: true`` with
``prompt_lookup_min``/``prompt_lookup_max``); ``config/speculative.py`` is
patched by patch-mtp-ngram-assist.sh to allow the flag with
``method='mtp'``. Without the flag it is never constructed.
"""

from __future__ import annotations

import os

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads


@njit(cache=True)
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
    out: np.ndarray,
) -> int:
    """Prompt-lookup KMP. Writes up to ``k`` proposed tokens into ``out`` and
    returns the count written (0 = no match / not a full run).

    This is the same reverse-KMP policy as the fork's standalone
    ``ngram_proposer.py``: prefer the longest matching suffix, and the earliest
    earlier occurrence.
    """
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return 0
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return 0

    tokens = origin_tokens[::-1]
    lps = np.zeros(max_ngram, dtype=np.int32)
    longest_ngram = 0
    position = 0
    prev_lps = 0
    i = 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_ngram:
        return 0
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    if k <= 0:
        return 0
    out[:k] = origin_tokens[start_position : start_position + k]
    return k


@njit(parallel=True)
def _batch_mtp_ngram(
    token_ids_cpu: np.ndarray,
    num_tokens_no_spec: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
    scan_cap: int,
    overlay_tokens: np.ndarray,
    overlay_full: np.ndarray,
) -> None:
    for i in prange(token_ids_cpu.shape[0]):
        n = int(num_tokens_no_spec[i])
        n = min(n, scan_cap)
        ctx = token_ids_cpu[i, :n]
        out = np.zeros(k, dtype=np.int32)
        cnt = _find_longest_matched_ngram_and_propose_tokens(
            ctx, min_ngram, max_ngram, max_model_len, k, out
        )
        if cnt == k:
            overlay_tokens[i, :k] = out[:k]
            overlay_full[i] = 1


class MtpNgramAssist:
    """Host-side prompt-ngram lookup adapter around the KMP kernel.

    The window comes from speculative_config (``prompt_lookup_min/max``,
    passed via --speculative-config): config/speculative.py is patched to
    allow ``ngram_assist`` with ``method='mtp'``, and the config keeps those
    values untouched for non-dflash methods when the flag is set.
    """

    def __init__(self, vllm_config) -> None:
        sc = vllm_config.speculative_config
        assert sc.prompt_lookup_min is not None
        assert sc.prompt_lookup_max is not None
        self.min_n = int(sc.prompt_lookup_min)
        self.max_n = int(sc.prompt_lookup_max)
        self.k = int(sc.num_speculative_tokens)
        self.max_model_len = vllm_config.model_config.max_model_len
        # Cap the per-request history scanned by the KMP. Repeats that pay off
        # are local; this bounds worst-case CPU on very long contexts. Raise
        # it via env to consider matches farther back in history.
        self.scan_cap = int(os.getenv("VLLM_SM70_MTP_NGRAM_SCAN_CAP", "16384"))
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.overlay_tokens = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self.overlay_full = np.zeros(max_num_seqs, dtype=np.int32)
        self._jit_ready = False

    def _ensure_jit(self) -> None:
        if self._jit_ready:
            return
        dummy = np.zeros((1, self.max_model_len), dtype=np.int32)
        lens = np.zeros(1, dtype=np.int32)
        _batch_mtp_ngram(
            dummy,
            lens,
            self.min_n,
            self.max_n,
            self.max_model_len,
            self.k,
            self.scan_cap,
            self.overlay_tokens,
            self.overlay_full,
        )
        self._jit_ready = True

    def propose(self, token_ids_cpu, num_tokens_no_spec):
        """Return ``[B, k]`` overlay tokens and a ``[B]`` int of full hits."""
        self._ensure_jit()
        B = token_ids_cpu.shape[0]
        if B == 0:
            return (
                np.zeros((0, self.k), dtype=np.int32),
                np.zeros(0, dtype=np.int32),
            )
        self.overlay_full[:B] = 0
        original_threads = get_num_threads()
        set_num_threads(1)
        try:
            _batch_mtp_ngram(
                token_ids_cpu,
                num_tokens_no_spec,
                self.min_n,
                self.max_n,
                self.max_model_len,
                self.k,
                self.scan_cap,
                self.overlay_tokens,
                self.overlay_full,
            )
        finally:
            set_num_threads(original_threads)
        return (
            self.overlay_tokens[:B].copy(),
            self.overlay_full[:B].copy(),
        )
