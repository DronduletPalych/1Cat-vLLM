# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore llama.cpp Qwen3.5 dense weights to vLLM checkpoint layout.

Name and GDN layout rules are adapted from vllm-gguf-plugin at
e2b8ad532b8b5ea175100202c30430c1d2b5e6a8 (Apache-2.0).
"""

import os

import gguf
import numpy as np
import regex as re
import torch

from vllm.model_executor.layers.quantization.gguf_layout import GGUFHeadTilingLayout
from vllm.transformers_utils.gguf_tensor_reader import dequantize, quant_size

_GLOBALS = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
_LAYERS = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_a.weight": "linear_attn.A_log",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

# llama.cpp ``clip`` mmproj layout -> Qwen3_VisionTransformer parameters.
# The names below are relative to ``model.visual.``; the VL model's
# WeightsMapper rewrites that prefix to the module's ``visual.`` at load time.
_VISION_BLOCKS = {
    "attn_qkv": "attn.qkv",  # fused Q/K/V; QKVParallelLinear shards it
    "attn_out": "attn.proj",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
    "ln1": "norm1",
    "ln2": "norm2",
}
_VISION_GLOBALS = {
    "mm.0.weight": "merger.linear_fc1.weight",
    "mm.0.bias": "merger.linear_fc1.bias",
    "mm.2.weight": "merger.linear_fc2.weight",
    "mm.2.bias": "merger.linear_fc2.bias",
    "v.post_ln.weight": "merger.norm.weight",
    "v.post_ln.bias": "merger.norm.bias",
    "v.position_embd.weight": "pos_embed.weight",
    "v.patch_embd.weight": "patch_embed.proj.weight",
    "v.patch_embd.bias": "patch_embed.proj.bias",
}
# llama.cpp splits the patch Conv3d kernel (temporal=2) into two time slices.
_VISION_PATCH_EMBED = "v.patch_embd.weight"
_VISION_PATCH_EMBED_TAIL = "v.patch_embd.weight.1"
_VISION_PREFIX = "model.visual."


def _embedding_chunks(tensor, dtype, name, rows_per_chunk=1024):
    rows, columns = map(int, tensor.shape[::-1])
    for start in range(0, rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, rows)
        data = gguf.quants.dequantize(tensor.data[start:stop], tensor.tensor_type)
        decoded = torch.from_numpy(data)
        converted = decoded.to(dtype)
        if torch.any(torch.isfinite(decoded) & ~torch.isfinite(converted)):
            raise ValueError(
                f"GGUF {name}: values overflow {dtype}; use --dtype float32"
            )
        yield start, stop, converted


def _decode_embedding_rows(tensor, dtype, name, start, stop):
    data = gguf.quants.dequantize(tensor.data[start:stop], tensor.tensor_type)
    decoded = torch.from_numpy(data)
    converted = decoded.to(dtype)
    if torch.any(torch.isfinite(decoded) & ~torch.isfinite(converted)):
        raise ValueError(f"GGUF {name}: values overflow {dtype}; use --dtype float32")
    return converted


def _validate_embedding_range(tensor, dtype, name, rows_per_chunk=1024):
    """Check that every decoded row fits ``dtype`` (the dense loader's range
    check) without keeping the table.

    The answer depends only on the file, so it is recorded next to vLLM's
    cache keyed by file identity and tensor offset; the first check decodes
    row chunks on a few threads.
    """
    import hashlib
    import json
    from concurrent.futures import ThreadPoolExecutor

    from vllm import envs

    base = tensor.data
    while base is not None and not isinstance(base, np.memmap):
        base = base.base
    marker = None
    if base is not None and base.filename is not None:
        path = str(base.filename)
        stat = os.stat(path)
        key = json.dumps(
            [
                path,
                stat.st_size,
                stat.st_mtime_ns,
                int(getattr(tensor, "data_offset", 0)),
                str(dtype),
            ]
        )
        marker = os.path.join(
            envs.VLLM_CACHE_ROOT,
            "gguf_range_checks",
            hashlib.sha256(key.encode()).hexdigest()[:32],
        )
        if os.path.exists(marker):
            return
    rows = int(tensor.shape[-1])
    workers = max(1, min(6, (os.cpu_count() or 1) // 4))
    with ThreadPoolExecutor(workers) as pool:
        # list() re-raises the first failing chunk in row order.
        list(
            pool.map(
                lambda start: _decode_embedding_rows(
                    tensor, dtype, name, start, min(start + rows_per_chunk, rows)
                ),
                range(0, rows, rows_per_chunk),
            )
        )
    if marker is not None:
        try:
            os.makedirs(os.path.dirname(marker), exist_ok=True)
            with open(marker, "w") as file:
                file.write(key)
        except OSError:
            pass


def _dequantize_embedding(tensor, dtype, name, rows_per_chunk=1024):
    """Bound temporary decode/range-check memory for dense vocabulary tables.

    Row chunks are independent; NumPy and Torch release the GIL inside each
    decode, so a few threads shorten startup without changing any value.
    """
    from concurrent.futures import ThreadPoolExecutor

    rows, columns = map(int, tensor.shape[::-1])
    weight = torch.empty((rows, columns), dtype=dtype, device="cpu")

    def fill(start):
        stop = min(start + rows_per_chunk, rows)
        weight[start:stop].copy_(
            _decode_embedding_rows(tensor, dtype, name, start, stop)
        )

    workers = max(1, min(6, (os.cpu_count() or 1) // 4))
    with ThreadPoolExecutor(workers) as pool:
        # list() re-raises the first failing chunk in row order.
        list(pool.map(fill, range(0, rows, rows_per_chunk)))
    return weight


class Qwen35Adapter:
    global_names = _GLOBALS
    layer_names = _LAYERS
    architecture_label = "Qwen3.5"
    packed_embeddings = True
    # Adapters that reuse this mapping for a text-only checkpoint (for example
    # the DFlash draft) keep the default: their tensor set has no mmproj side.
    multimodal = False

    def __init__(self, config, tp_size=1, multimodal=False):
        self.config = config
        self.tp_size = tp_size
        self.multimodal = multimodal
        self.fallback_reasons = {}
        repeat, remainder = divmod(
            config.linear_num_value_heads, config.linear_num_key_heads
        )
        if remainder or repeat < 1:
            raise ValueError("GGUF GDN value heads must be a multiple of key heads")
        self.layout = (
            GGUFHeadTilingLayout(repeat, config.linear_value_head_dim)
            if repeat > 1
            else None
        )

    def _map_vision_name(self, name):
        """Map a llama.cpp ``clip`` mmproj tensor onto ``model.visual.*``.

        Returns None for names that are not vision tensors. The mmproj stores a
        fused ``attn_qkv`` (QKVParallelLinear shards it on load), pre-merge
        ``v.post_ln`` for the merger norm, and a two-layer MLP projector.
        """
        if not self.multimodal:
            return None
        if (global_name := _VISION_GLOBALS.get(name)) is not None:
            return _VISION_PREFIX + global_name
        match = re.fullmatch(
            r"v\.blk\.(\d+)\.(attn_qkv|attn_out|ffn_up|ffn_down|ln1|ln2)"
            r"\.(weight|bias)",
            name,
        )
        if match:
            return (
                f"{_VISION_PREFIX}blocks.{int(match[1])}."
                f"{_VISION_BLOCKS[match[2]]}.{match[3]}"
            )
        return None

    def _is_vision_name(self, name):
        """Vision tensors are dense F16/F32 parameters, never GGUF qweights."""
        return self.multimodal and name.startswith(_VISION_PREFIX)

    def build_name_map(self, tensors):
        result = {}

        def _text_name(base):
            # In multimodal mode the Qwen3_5ForConditionalGeneration mapper
            # routes `model.language_model.` -> `language_model.model.` and
            # `model.visual.` -> `visual.`, while the text-only
            # Qwen3_5ForCausalLM keeps the flat `model.*` layout. Only the
            # leading `model.` is re-rooted; `lm_head.weight` (no `model.`
            # prefix) is left untouched for the mapper's `lm_head.` rule.
            if self.multimodal and base.startswith("model."):
                return "model.language_model." + base[len("model.") :]
            return base

        for name in tensors:
            if name in self.global_names:
                result[name] = _text_name(self.global_names[name])
                continue
            vision_name = self._map_vision_name(name)
            if vision_name is not None:
                result[name] = vision_name
                continue
            if self.multimodal:
                if name == _VISION_PATCH_EMBED_TAIL:
                    # Second temporal slice of the patch Conv3d kernel; it is
                    # stacked onto v.patch_embd.weight when yielding weights.
                    continue
                if name.startswith(("v.", "mm.")):
                    raise ValueError(
                        f"Unmapped {self.architecture_label} mmproj tensor: {name}"
                    )
            match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
            if match:
                block, suffix = int(match[1]), match[2]
                if block >= self.config.num_hidden_layers:
                    # The target does not execute nextn. Its draft adapter has a
                    # separate mapping and must consume these tensors explicitly.
                    if block < self.config.num_hidden_layers + getattr(
                        self.config, "num_nextn_predict_layers", 0
                    ):
                        continue
                elif suffix in self.layer_names:
                    result[name] = _text_name(
                        f"model.layers.{block}.{self.layer_names[suffix]}"
                    )
                    continue
            raise ValueError(f"Unmapped {self.architecture_label} GGUF tensor: {name}")
        return result

    def is_linear(self, name):
        if self._is_vision_name(name):
            return False
        return name.endswith(".weight") and not name.endswith(
            ("norm.weight", "conv1d.weight", "embed_tokens.weight")
        )

    def needs_dense_fallback(self, name, tensor):
        if not name.endswith(
            (".down_proj.weight", ".out_proj.weight", ".o_proj.weight")
        ):
            return False
        block_size, _ = quant_size(tensor.tensor_type)
        local_k, remainder = divmod(int(tensor.shape[0]), self.tp_size)
        tiled_span = local_k
        if self.layout is not None and name.endswith("linear_attn.out_proj.weight"):
            tiled_span //= self.layout.heads_per_group
        if remainder or local_k % block_size or tiled_span % block_size:
            self.fallback_reasons[name.removesuffix(".weight")] = (
                f"dequantize_to_fp16_before_tp:local_K={local_k}, "
                f"stored_tile_K={tiled_span}, GGML_block={block_size}"
            )
            return True
        return False

    def linear_layouts(self, name_map):
        if self.layout is None:
            return {}
        return {
            name.removesuffix(".weight"): self.layout
            for name in name_map.values()
            if name.endswith("linear_attn.out_proj.weight")
        }

    def restore(self, name, weight):
        if name.endswith(".A_log"):
            if not torch.all(weight < 0):
                raise ValueError(f"GGUF {name} must contain negative exp(A_log)")
            weight = torch.log(-weight.float())
        if self.layout is not None:
            layout = self.layout
            if name.endswith(
                (".A_log", ".dt_bias", "in_proj_a.weight", "in_proj_b.weight")
            ):
                weight = layout.weight_to_vllm(weight, dim=0, head_dim=1)
            elif name.endswith("in_proj_z.weight"):
                weight = layout.weight_to_vllm(weight, dim=0)
            elif name.endswith(("in_proj_qkv.weight", "conv1d.weight")):
                qk = (
                    self.config.linear_key_head_dim
                    * self.config.linear_num_key_heads
                    * 2
                )
                weight = torch.cat(
                    [weight[:qk], layout.weight_to_vllm(weight[qk:], dim=0)]
                )
        if name.endswith("norm.weight") and not name.endswith(
            "linear_attn.norm.weight"
        ):
            weight = weight - 1
        if name.endswith("conv1d.weight"):
            weight = weight.unsqueeze(1)
        return weight

    def weights(self, tensors, name_map, dtype):
        # All linear type descriptors must arrive before any weight payload,
        # including F16/F32 shards mixed with packed GGML shards.
        for raw, name in name_map.items():
            packed_embedding = (
                self.packed_embeddings
                and not getattr(self.config, "tie_word_embeddings", False)
                and name.endswith("embed_tokens.weight")
                and tensors[raw].tensor_type
                not in (
                    gguf.GGMLQuantizationType.F32,
                    gguf.GGMLQuantizationType.F16,
                    gguf.GGMLQuantizationType.BF16,
                )
            )
            if self.is_linear(name) or packed_embedding:
                yield (
                    name.removesuffix(".weight") + ".qweight_type",
                    torch.tensor(
                        int(gguf.GGMLQuantizationType.F16)
                        if self.needs_dense_fallback(name, tensors[raw])
                        else int(tensors[raw].tensor_type)
                    ),
                )
        for raw, name in name_map.items():
            if raw == _VISION_PATCH_EMBED and self._is_vision_name(name):
                # llama.cpp stores the patch Conv3d kernel (temporal=2) as two
                # time slices; gguf_reader already returns them in torch order
                # (out, in, kh, kw), so they only need joining on the temporal
                # axis. No transpose is required.
                slices = [
                    torch.from_numpy(tensors[part].data.copy())
                    for part in (_VISION_PATCH_EMBED, _VISION_PATCH_EMBED_TAIL)
                ]
                yield name, torch.stack(slices, dim=2).to(dtype)
                continue
            tensor = tensors[raw]
            quantized = tensor.tensor_type not in (
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.BF16,
            )
            if quantized and name.endswith("embed_tokens.weight"):
                if self.packed_embeddings and not getattr(
                    self.config, "tie_word_embeddings", False
                ):
                    # Validate the same FP16 range as the dense loader, using
                    # only a bounded row chunk. Serving decodes selected rows
                    # from the original GGML bytes through GGUFEmbeddingMethod.
                    _validate_embedding_range(tensor, dtype, raw)
                    yield (
                        name.removesuffix(".weight") + ".qweight",
                        torch.from_numpy(tensor.data.copy()),
                    )
                    continue
                # Embedding row order is unchanged by restoration. Keep the
                # existing global-table contract for the TP weight loader,
                # without full FP32 decode and boolean temporary tables.
                yield name, _dequantize_embedding(tensor, dtype, raw)
                continue
            dense_fallback = self.needs_dense_fallback(name, tensor)
            if quantized and (not self.is_linear(name) or dense_fallback):
                # Embeddings and convolution use the model's dense parameters.
                data = dequantize(tensor.data, tensor.tensor_type)
                # Dequantization owns this dense array; avoid a second copy.
                weight = torch.from_numpy(data)
            elif tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
                data = tensor.data.view(np.uint16).copy()
                weight = torch.from_numpy(data).view(torch.bfloat16)
            else:
                weight = torch.from_numpy(tensor.data.copy())
            if not self._is_vision_name(name):
                weight = self.restore(name, weight)
            if not quantized or dense_fallback or not self.is_linear(name):
                # The model keeps A_log in FP32 even under --dtype half.
                # Preserve inverse-log precision rather than rounding through
                # FP16 before the destination parameter copies it back to FP32.
                target_dtype = torch.float32 if name.endswith(".A_log") else dtype
                converted = weight.to(target_dtype)
                if torch.any(torch.isfinite(weight) & ~torch.isfinite(converted)):
                    raise ValueError(
                        f"GGUF {raw}: values overflow {dtype}; use --dtype float32"
                    )
                weight = converted
            if self.is_linear(name):
                name = name.removesuffix(".weight") + ".qweight"
            yield name, weight
