# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import fcntl
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

logger = init_logger(__name__)

# Contents self-check: remember a cheap fingerprint of every CPU page written by
# a store and verify it again when that page is read back by a load. This
# separates "the bytes were corrupted in flight" from "the load reads a page
# that was never written for this key".
SELFCHECK = os.environ.get("VLLM_KV_OFFLOAD_SELFCHECK", "") in ("1", "true", "yes")


def page_signature(tensor: torch.Tensor, offset: int, size: int):
    """Cheap order-sensitive fingerprint of ``size`` bytes at ``offset``.

    Canonical CPU tensors are strided views into a shared interleaved region
    whose row stride is ``page_size * world_size``, so the row is derived from
    the tensor stride instead of its column count.  Returns ``None`` when the
    slice is not addressable through this view.
    """
    stride = tensor.stride(0)
    row = offset // stride
    col = offset % stride
    if row >= tensor.shape[0] or col + size > tensor.shape[1]:
        return None
    view = tensor[row, col : col + size]
    return (
        int(view.sum(dtype=torch.int64).item()),
        int(view[::7].sum(dtype=torch.int64).item()),
        int(view[3::11].sum(dtype=torch.int64).item()),
        int(torch.count_nonzero(view).item()),
    )


@dataclass
class Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
):
    """
    Compute byte pointers for sub-blocks of the given block IDs.

    Each block in block_ids contains block_size_factor sub-blocks.
    The pointer for sub-block j of block b is:
        base_ptr + b * row_stride + j * sub_block_size

    where sub_block_size = tensor.shape[1] // block_size_factor (gpu page size).

    This handles tensors where row_stride != block_size_factor * sub_block_size
    (e.g. non-contiguous CPU tensors).

    Args:
        block_ids: array of block IDs at the tensor's native granularity.
        block_size_factor: number of sub-blocks per block.
        output: pre-allocated int64 array to write pointers into.
        tensor: the source or destination tensor.
        skip_count: sub-blocks to skip in the first block.
    """
    assert skip_count < block_size_factor

    num_sub_blocks = len(output)
    base_ptr = tensor.data_ptr()
    row_stride = tensor.stride(0)

    if block_size_factor == 1:
        # Fast path: 1:1 mapping, no sub-block expansion needed.
        output[:] = base_ptr + block_ids[:num_sub_blocks] * row_stride
        return

    # Vectorized expansion for block_size_factor > 1.
    assert tensor.shape[1] % block_size_factor == 0
    sub_block_size = tensor.shape[1] // block_size_factor
    sub_offsets = np.arange(block_size_factor, dtype=np.int64) * sub_block_size
    # (num_blocks, 1) + (1, block_size_factor) -> (num_blocks, block_size_factor)
    all_ptrs = (
        base_ptr + block_ids.astype(np.int64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    # Flatten and apply skip_count / truncation
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


def pin_mmap_region(region: SharedOffloadRegion) -> None:
    """Register the entire mmap as CUDA pinned memory via cudaHostRegister."""
    rank = region.rank

    base_ptr = region._base.data_ptr()
    # Long-term pinning can migrate pages. Concurrent registrations of the
    # same shared region can hold pins that prevent another worker's migration
    # and make pin_user_pages return ENOMEM despite sufficient free RAM.
    # Each worker opens this file independently, so flock serializes registration
    # across processes without keeping a lock during KV transfers.
    assert region.fd is not None
    fcntl.flock(region.fd, fcntl.LOCK_EX)
    try:
        result = torch.cuda.cudart().cudaHostRegister(
            base_ptr, region.total_size_bytes, 0
        )
    finally:
        fcntl.flock(region.fd, fcntl.LOCK_UN)
    if result.value != 0:
        # The batch transfer path requires registered host memory. Continuing
        # also leaves the CUDA error pending for an unrelated later kernel.
        raise RuntimeError(
            f"cudaHostRegister failed for rank={rank}, "
            f"region={region.mmap_path}, bytes={region.total_size_bytes} "
            f"(code={result.value}); KV batch transfers require pinned memory"
        )
    else:
        logger.debug(
            "cudaHostRegister rank=%d %.2f GB",
            rank,
            region.total_size_bytes / 1e9,
        )
        region.is_pinned = True


class SingleDirectionOffloadingHandler(OffloadingHandler):
    """
    SingleDirectionOffloadingHandler handles transfers for a single direction,
    either CPU->GPU or GPU->CPU.
    Transfers are guaranteed to be executed in order of their submission.
    Each transfer uses a unique CUDA stream, and its stream will start
    executing only after the streams of previous transfers have finished.
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        block_size_factor: int,
        kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        mmap_region: SharedOffloadRegion | None = None,
        group_mmap_regions: dict[int, SharedOffloadRegion] | None = None,
        selfcheck_store: dict | None = None,
        tensor_to_group: dict[int, int] | None = None,
    ):
        """
        Initialize a SingleDirectionOffloadingHandler.

        Args:
            gpu_tensors: list of GPU KV cache tensors.
                Each of shape (num_gpu_blocks, gpu_page_size_bytes) with dtype int8.
            cpu_tensors: list of CPU KV cache tensors.
                Each of shape (num_cpu_blocks, cpu_page_size_bytes) with dtype int8.
                Order should match gpu_tensors.
            kv_cache_groups_data_refs: list of CanonicalKVCacheRef per group.
            gpu_to_cpu: if True, transfer from GPU to CPU; otherwise CPU to GPU.
        """
        assert len(gpu_tensors) == len(cpu_tensors)
        assert len(gpu_tensors) > 0

        # assert input tensors are as expected
        for gpu_tensor, cpu_tensor in zip(gpu_tensors, cpu_tensors):
            assert gpu_tensor.dtype == torch.int8
            assert gpu_tensor.ndim == 2
            assert gpu_tensor.is_cuda
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            _, gpu_page_size = gpu_tensor.shape
            _, cpu_page_size = cpu_tensor.shape
            assert cpu_page_size == gpu_page_size * block_size_factor

        self.src_tensors: list[torch.Tensor] = (
            gpu_tensors if gpu_to_cpu else cpu_tensors
        )
        self.dst_tensors: list[torch.Tensor] = (
            cpu_tensors if gpu_to_cpu else gpu_tensors
        )
        self.gpu_to_cpu: bool = gpu_to_cpu
        self.kv_cache_groups_data_refs = kv_cache_groups_data_refs

        # GPU blocks may be smaller
        # cpu_page_size = gpu_page_size * block_size_factor.
        self.src_block_size_factor = 1 if self.gpu_to_cpu else block_size_factor
        self.dst_block_size_factor = block_size_factor if self.gpu_to_cpu else 1

        self.transfer_type = ("GPU", "CPU") if self.gpu_to_cpu else ("CPU", "GPU")
        # mmap_region to clean up on shutdown (gpu_to_cpu handler owns it)
        self._mmap_region = mmap_region
        self._group_mmap_regions = dict(group_mmap_regions or {})
        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        # queue of transfers (job_id, stream, event)
        self._transfers: deque[Transfer] = deque()
        # list of CUDA streams available for re-use
        self._stream_pool: list[torch.cuda.Stream] = []
        # list of CUDA events available for re-use
        self._event_pool: list[torch.Event] = []
        # contents self-check state
        self._selfcheck = SELFCHECK
        self._selfcheck_store: dict = (
            selfcheck_store if selfcheck_store is not None else {}
        )
        self._selfcheck_jobs: dict[int, tuple] = {}
        self._tensor_to_group = tensor_to_group or {}
        if self._selfcheck:
            logger.info(
                "OFFLOAD-XCHK INIT g2c=%s factor=%d n_tensors=%d n_groups=%d "
                "shapes=%s refcounts=%s",
                gpu_to_cpu,
                block_size_factor,
                len(gpu_tensors),
                len(kv_cache_groups_data_refs),
                [(int(t.shape[0]), int(t.shape[1])) for t in gpu_tensors[:3]],
                [len(r) for r in kv_cache_groups_data_refs],
            )

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        num_src_blocks = len(src_blocks)
        num_dst_blocks = len(dst_blocks)

        # There are 2 types of transfers:
        # 1. GPU -> CPU
        # 2. CPU -> GPU
        #
        # transfers are also to CPU blocks, EXCEPT MAYBE for the first and last block.
        # i.e. the first and last CPU blocks in src_blocks can match against
        # a smaller (byte-wise) set of GPU blocks in dst_blocks.
        # In such cases, we may need to skip some gpu-sized sub-blocks,
        # and start reading/writing from the middle of the first CPU block.
        # If we have multiple KV cache groups (when using HMA with hybrid models),
        # we may have a partial first/last CPU block per each group.
        # The group_sizes parameter encodes the size of each group of blocks
        # in the GPU dst_blocks.
        # If group_sizes is None, we assume all blocks belong to a single group.
        # The logical_offset parameter maps each group of blocks to its logical
        # offset inside the request, counting in GPU blocks.
        # This allows us to find the correct starting position
        # in the matching first CPU block.

        # extract group_sizes from the GPU spec
        gpu_spec = src_spec if self.gpu_to_cpu else dst_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        group_sizes = gpu_spec.group_sizes
        assert len(group_sizes) == len(self.kv_cache_groups_data_refs)

        # extract block indices from the GPU spec
        block_indices = gpu_spec.block_indices
        assert len(block_indices) == len(self.kv_cache_groups_data_refs)

        num_copy_ops = 0
        for group_size, group_data_refs in zip(
            group_sizes, self.kv_cache_groups_data_refs
        ):
            num_copy_ops += group_size * len(group_data_refs)

        all_src = np.empty(num_copy_ops, dtype=np.int64)
        all_dst = np.empty(num_copy_ops, dtype=np.int64)
        all_sizes = np.empty(num_copy_ops, dtype=np.int64)
        all_tidx = np.empty(num_copy_ops, dtype=np.int64)

        src_offset = 0
        dst_offset = 0
        op_idx = 0
        # count total number of bytes copied
        num_transfer_bytes = 0
        for grp_i, (group_size, block_idx, group_data_refs) in enumerate(
            zip(group_sizes, block_indices, self.kv_cache_groups_data_refs)
        ):
            if group_size == 0:
                continue

            src_logical_blocks_to_skip = block_idx % self.src_block_size_factor
            dst_logical_blocks_to_skip = block_idx % self.dst_block_size_factor
            src_logical_blocks_count = group_size + src_logical_blocks_to_skip
            dst_logical_blocks_count = group_size + dst_logical_blocks_to_skip

            dst_blocks_count = cdiv(
                dst_logical_blocks_count, self.dst_block_size_factor
            )
            dst_end_offset = dst_offset + dst_blocks_count
            assert dst_end_offset <= num_dst_blocks

            src_blocks_count = cdiv(
                src_logical_blocks_count, self.src_block_size_factor
            )
            src_end_offset = src_offset + src_blocks_count
            assert src_end_offset <= num_src_blocks

            group_src = src_blocks[src_offset:src_end_offset]
            group_dst = dst_blocks[dst_offset:dst_end_offset]

            if (
                self._selfcheck
                and os.environ.get("VLLM_OFFLOAD_XCHK_GRP") == "1"
                and group_size <= 8
            ):
                logger.info(
                    "OFFLOAD-XCHK GRP job=%d dir=%s g=%d size=%d idx=%d "
                    "src=%s dst=%s tidx=%s",
                    job_id,
                    "G2C" if self.gpu_to_cpu else "C2G",
                    grp_i,
                    group_size,
                    block_idx,
                    list(group_src[:6]),
                    list(group_dst[:6]),
                    [r.tensor_idx for r in group_data_refs],
                )

            for data_ref in group_data_refs:
                t_idx = data_ref.tensor_idx
                end_idx = op_idx + group_size

                compute_sub_block_ptrs(
                    group_src,
                    self.src_block_size_factor,
                    all_src[op_idx:end_idx],
                    self.src_tensors[t_idx],
                    skip_count=src_logical_blocks_to_skip,
                )
                compute_sub_block_ptrs(
                    group_dst,
                    self.dst_block_size_factor,
                    all_dst[op_idx:end_idx],
                    self.dst_tensors[t_idx],
                    skip_count=dst_logical_blocks_to_skip,
                )

                all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
                all_tidx[op_idx:end_idx] = t_idx
                num_transfer_bytes += group_size * data_ref.page_size_bytes
                op_idx = end_idx

            src_offset = src_end_offset
            dst_offset = dst_end_offset

        assert src_offset == num_src_blocks
        assert dst_offset == num_dst_blocks
        assert op_idx == num_copy_ops

        batch_src = torch.from_numpy(all_src)
        batch_dst = torch.from_numpy(all_dst)
        batch_sizes = torch.from_numpy(all_sizes)

        stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

        # D2H must wait for computation to finish writing the source. H2D
        # must also wait: the compute stream may still be zeroing freshly
        # allocated destination blocks. Otherwise that zeroing can overwrite
        # restored cache data after the transfer completes.
        stream.wait_stream(torch.cuda.current_stream())
        if self._transfers:
            last_transfer: Transfer = self._transfers[-1]
            last_event = last_transfer.end_event
            # assure job will start only after the previous one completes
            stream.wait_event(last_event)
        # CPU->GPU reads from host pinned memory, which is never written
        # by a concurrent GPU stream, so CU_MEMCPY_SRC_ACCESS_ORDER_ANY is
        # safe and lets the driver pipeline source reads. GPU->CPU reads
        # from the live GPU KV cache, which the compute stream keeps
        # writing; we must keep STREAM ordering so source reads are gated
        # by the transfer stream's wait_stream(compute) barrier.
        is_src_access_order_any = not self.gpu_to_cpu
        with torch.cuda.stream(stream):
            start_event.record(stream)
            if num_copy_ops > 0:
                ops.swap_blocks_batch(
                    batch_src,
                    batch_dst,
                    batch_sizes,
                    is_src_access_order_any=is_src_access_order_any,
                )
            end_event.record(stream)

        self._transfer_events[job_id] = end_event
        if self._selfcheck:
            self._selfcheck_jobs[job_id] = (
                all_tidx.copy(),
                all_src.copy(),
                all_dst.copy(),
                all_sizes.copy(),
            )
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_transfer_bytes,
            )
        )

        # success
        return True

    def _run_selfcheck(self, transfer: Transfer) -> None:
        """Verify CPU page contents against what the matching store wrote."""
        meta = self._selfcheck_jobs.pop(transfer.job_id, None)
        if meta is None:
            return
        tidx_arr, src_arr, dst_arr, sizes = meta
        torch.cuda.current_stream().wait_event(transfer.end_event)
        per_group: dict[int, list[int]] = {}
        misses = 0
        mismatches = 0
        zero_pages = 0
        zero_loaded = 0
        overwrites = 0
        cpu_tensors = self.dst_tensors if self.gpu_to_cpu else self.src_tensors
        ptr_arr = dst_arr if self.gpu_to_cpu else src_arr
        uniq_cpu = len(set(int(x) for x in ptr_arr))
        dup_ptr = len(ptr_arr) - uniq_cpu
        if len(sizes) <= 64 and os.environ.get("VLLM_OFFLOAD_XCHK_PTRS") == "1":
            compact = [
                (
                    int(tidx_arr[i]),
                    int(ptr_arr[i]) - cpu_tensors[int(tidx_arr[i])].data_ptr(),
                )
                for i in range(len(sizes))
            ]
            logger.warning(
                "OFFLOAD-XCHK PTRS job=%d dir=%s n=%d uniq=%d %s",
                transfer.job_id,
                "G2C" if self.gpu_to_cpu else "C2G",
                len(sizes),
                uniq_cpu,
                compact,
            )
        for i in range(len(sizes)):
            t_idx = int(tidx_arr[i])
            size = int(sizes[i])
            group = self._tensor_to_group.get(t_idx, -1)
            acc = per_group.setdefault(group, [0, 0])
            acc[0] += 1
            acc[1] += size
            if self.gpu_to_cpu:
                cpu_t = self.dst_tensors[t_idx]
                off = int(dst_arr[i]) - cpu_t.data_ptr()
                sig = page_signature(cpu_t, off, size)
                if sig is None:
                    continue
                prev = self._selfcheck_store.get((t_idx, off))
                if prev is not None and prev != sig:
                    overwrites += 1
                    if overwrites <= 8:
                        logger.warning(
                            "OFFLOAD-XCHK OVERWRITE group=%d tensor=%d "
                            "cpu_off=%d size=%d prev=%s new=%s",
                            group,
                            t_idx,
                            off,
                            size,
                            prev,
                            sig,
                        )
                self._selfcheck_store[(t_idx, off)] = sig
                if sig[3] == 0:
                    zero_pages += 1
                    if zero_pages <= 6:
                        logger.warning(
                            "OFFLOAD-XCHK STORE-ZERO group=%d tensor=%d "
                            "cpu_off=%d size=%d",
                            group,
                            t_idx,
                            off,
                            size,
                        )
            else:
                cpu_t = self.src_tensors[t_idx]
                off = int(src_arr[i]) - cpu_t.data_ptr()
                gpu_t = self.dst_tensors[t_idx]
                goff = int(dst_arr[i]) - gpu_t.data_ptr()
                expected = self._selfcheck_store.get((t_idx, off))
                actual = page_signature(gpu_t, goff, size)
                if actual is None:
                    continue
                if actual[3] == 0:
                    zero_loaded += 1
                    if zero_loaded <= 6:
                        logger.warning(
                            "OFFLOAD-XCHK LOAD-ZERO group=%d tensor=%d "
                            "cpu_off=%d size=%d known=%s",
                            group,
                            t_idx,
                            off,
                            size,
                            expected is not None,
                        )
                if expected is None:
                    misses += 1
                    if misses <= 8:
                        logger.warning(
                            "OFFLOAD-XCHK MISS group=%d tensor=%d cpu_off=%d "
                            "size=%d gpu_nonzero=%d",
                            group,
                            t_idx,
                            off,
                            size,
                            actual[3],
                        )
                elif expected != actual:
                    mismatches += 1
                    if mismatches <= 8:
                        logger.warning(
                            "OFFLOAD-XCHK MISMATCH group=%d tensor=%d "
                            "cpu_off=%d size=%d gpu_nonzero=%d expected=%s "
                            "actual=%s",
                            group,
                            t_idx,
                            off,
                            size,
                            actual[3],
                            expected,
                            actual,
                        )
        logger.info(
            "OFFLOAD-XCHK job=%d dir=%s ops=%d uniq_cpu=%d dup_ptr=%d groups=%s "
            "misses=%d mismatches=%d known=%d zero_src=%d zero_dst=%d owr=%d",
            transfer.job_id,
            "G2C" if self.gpu_to_cpu else "C2G",
            len(sizes),
            uniq_cpu,
            dup_ptr,
            sorted((g, v[0], v[1]) for g, v in per_group.items()),
            misses,
            mismatches,
            len(self._selfcheck_store),
            zero_pages,
            zero_loaded,
            overwrites,
        )

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            transfer = self._transfers.popleft()
            if self._selfcheck:
                self._run_selfcheck(transfer)
            transfer_time = (
                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            )  # elapsed_time is in milliseconds
            result = TransferResult(
                job_id=transfer.job_id,
                success=True,
                transfer_size=transfer.num_bytes,
                transfer_time=transfer_time,
                transfer_type=self.transfer_type,
            )

            results.append(result)
            self._stream_pool.append(transfer.stream)
            self._event_pool.append(transfer.end_event)
            self._event_pool.append(transfer.start_event)
            del self._transfer_events[transfer.job_id]
        return results

    def wait(self, job_ids: set[int]):
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    def shutdown(self) -> None:
        while self._transfers:
            transfer = self._transfers.popleft()
            transfer.end_event.synchronize()
        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()
        if self._mmap_region is not None:
            self._mmap_region.cleanup()
            self._mmap_region = None
        for region in self._group_mmap_regions.values():
            region.cleanup()
        self._group_mmap_regions.clear()


def partition_kv_caches(
    kv_caches: CanonicalKVCaches,
    group_page_sizes: dict[int, int],
    block_size_factor: int,
) -> CanonicalKVCaches:
    """Give each group private CPU backing while retaining its GPU views.

    Canonical GPU tensors can be shared across groups. Duplicate only their
    references here: CPU allocations must not alias across those group pools.
    """
    tensors: list[CanonicalKVCacheTensor] = []
    group_refs: list[list[CanonicalKVCacheRef]] = []
    for group_idx, refs in enumerate(kv_caches.group_data_refs):
        new_refs: list[CanonicalKVCacheRef] = []
        tensor_indices: dict[int, int] = {}
        allocated_page_size = 0
        if group_idx in group_page_sizes:
            for ref in refs:
                if ref.tensor_idx not in tensor_indices:
                    tensor_indices[ref.tensor_idx] = len(tensors)
                    tensor = kv_caches.tensors[ref.tensor_idx]
                    tensors.append(tensor)
                    allocated_page_size += tensor.page_size_bytes * block_size_factor
                new_refs.append(
                    CanonicalKVCacheRef(
                        tensor_idx=tensor_indices[ref.tensor_idx],
                        page_size_bytes=ref.page_size_bytes,
                    )
                )
            assert allocated_page_size <= group_page_sizes[group_idx], (
                f"CPU group {group_idx} exceeds its byte budget: "
                f"{allocated_page_size} > {group_page_sizes[group_idx]}"
            )
        # Scratch groups can have canonical GPU refs, but never receive an
        # offload key. Keep their group position with no CPU allocation.
        group_refs.append(new_refs)
    return CanonicalKVCaches(tensors=tensors, group_data_refs=group_refs)


class CpuGpuOffloadingHandlers:
    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
        mmap_region: SharedOffloadRegion | None = None,
        group_page_sizes: dict[int, int] | None = None,
        group_mmap_regions: dict[int, SharedOffloadRegion] | None = None,
        group_num_blocks: dict[int, int] | None = None,
    ):
        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        cpu_tensor: torch.Tensor | None = None
        # Group pools may hold different slot counts (e.g. strided Mamba
        # checkpoints). Private tensors belong to exactly one group.
        tensor_num_blocks: dict[int, int] = {}
        try:
            if group_page_sizes is not None:
                assert mmap_region is None, (
                    "Grouped pools require per-group mmap regions"
                )
                kv_caches = partition_kv_caches(
                    kv_caches, group_page_sizes, block_size_factor
                )
                for group_idx, refs in enumerate(kv_caches.group_data_refs):
                    slots = (group_num_blocks or {}).get(group_idx, num_cpu_blocks)
                    for ref in refs:
                        tensor_num_blocks[ref.tensor_idx] = slots
            else:
                assert group_num_blocks is None
            tensor_regions: dict[int, SharedOffloadRegion] = {}
            if group_mmap_regions is not None:
                assert group_page_sizes is not None
                assert set(group_mmap_regions) == set(group_page_sizes)
                for group_idx, region in group_mmap_regions.items():
                    assert region.num_blocks == (group_num_blocks or {}).get(
                        group_idx, num_cpu_blocks
                    )
                    assert region.rank is not None
                    assert (
                        region._worker_area_end - region._worker_offset
                        == (group_page_sizes[group_idx])
                    )
                    for ref in kv_caches.group_data_refs[group_idx]:
                        tensor_regions[ref.tensor_idx] = region
            pin_memory = is_pin_memory_available()
            logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
            self._mmap_region = mmap_region
            if mmap_region is not None and pin_memory:
                pin_mmap_region(mmap_region)

            if pin_memory and group_mmap_regions:
                for region in group_mmap_regions.values():
                    pin_mmap_region(region)

            for tensor_idx, kv_cache_tensor in enumerate(kv_caches.tensors):
                gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
                gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                    (-1, gpu_page_size_bytes)
                )
                cpu_page_size_bytes = gpu_page_size_bytes * block_size_factor

                tensor_region = tensor_regions.get(tensor_idx, mmap_region)
                if tensor_region is not None:
                    cpu_tensor = tensor_region.create_next_view(cpu_page_size_bytes)
                else:
                    t0 = time.monotonic()
                    tensor_slots = tensor_num_blocks.get(tensor_idx, num_cpu_blocks)
                    cpu_tensor = torch.zeros(
                        (tensor_slots, cpu_page_size_bytes),
                        dtype=torch.int8,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    logger.debug(
                        "torch.zeros pinned tensor %d×%d (%.2f GB): %.3f s",
                        tensor_slots,
                        cpu_page_size_bytes,
                        tensor_slots * cpu_page_size_bytes / 1e9,
                        time.monotonic() - t0,
                    )

                gpu_tensors.append(gpu_tensor)
                cpu_tensors.append(cpu_tensor)

            tensor_to_group: dict[int, int] = {}
            for group_idx, refs in enumerate(kv_caches.group_data_refs):
                for ref in refs:
                    tensor_to_group.setdefault(ref.tensor_idx, group_idx)
            selfcheck_store: dict = {}

            self.gpu_to_cpu_handler = SingleDirectionOffloadingHandler(
                gpu_tensors=gpu_tensors,
                cpu_tensors=cpu_tensors,
                block_size_factor=block_size_factor,
                kv_cache_groups_data_refs=kv_caches.group_data_refs,
                gpu_to_cpu=True,
                mmap_region=mmap_region,
                group_mmap_regions=group_mmap_regions,
                selfcheck_store=selfcheck_store,
                tensor_to_group=tensor_to_group,
            )

            self.cpu_to_gpu_handler = SingleDirectionOffloadingHandler(
                gpu_tensors=gpu_tensors,
                cpu_tensors=cpu_tensors,
                block_size_factor=block_size_factor,
                kv_cache_groups_data_refs=kv_caches.group_data_refs,
                gpu_to_cpu=False,
                selfcheck_store=selfcheck_store,
                tensor_to_group=tensor_to_group,
            )
        except Exception:
            # No transfers can have been submitted before construction returns.
            # Drop tensor views before releasing shared mappings and registration.
            cpu_tensor = None
            cpu_tensors.clear()
            gpu_tensors.clear()
            if mmap_region is not None:
                mmap_region.cleanup()
            for region in (group_mmap_regions or {}).values():
                region.cleanup()
            raise