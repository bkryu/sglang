from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.batch_invariant_quant import (
    flashinfer_bi_quant_device_ok,
    use_flashinfer_bi_quantizers,
)
from sglang.srt.layers.quantization.mxfp8_input import batch_invariant_or_deterministic
from sglang.srt.runtime_context import (
    get_exec,
    get_platform,
)
from sglang.srt.utils.custom_op import register_custom_op_from_extern
from sglang.srt.utils.device import get_device_capability, is_cuda
from sglang.srt.utils.log_utils import print_info_once

logger = logging.getLogger(__name__)


def use_flashinfer_bi_nvfp4(device: torch.device) -> bool:
    """Whether the deterministic mode runs NVFP4 activation quantization and the
    dense NVFP4 GEMM on FlashInfer's batch-invariant kernels on ``device``. With
    ``FLASHINFER_NVFP4_4OVER6`` set every site keeps its current kernel: the
    batch-invariant kernels do not implement the 4over6 recipe."""
    if not use_flashinfer_bi_quantizers(device):
        return False
    if envs.FLASHINFER_NVFP4_4OVER6.get():
        return False
    print_info_once(
        "Deterministic mode: NVFP4 activation quantization and the dense NVFP4 "
        "GEMM run FlashInfer's batch-invariant kernels."
    )
    return True


_FALLBACKS_LOGGED: set = set()


def _log_nvfp4_fallback_once(what: str, device: torch.device) -> None:
    """One log line per shape when a site under the deterministic mode keeps its
    previous kernel; nothing on later calls."""
    if what in _FALLBACKS_LOGGED or not batch_invariant_or_deterministic():
        return
    _FALLBACKS_LOGGED.add(what)
    if not envs.SGLANG_ENABLE_DETERMINISTIC_FLASHINFER_BI_KERNELS.get():
        return
    if envs.FLASHINFER_NVFP4_4OVER6.get():
        reason = "FLASHINFER_NVFP4_4OVER6 is set (no 4over6 recipe in the batch-invariant kernels)"
    elif not flashinfer_bi_quant_device_ok(device):
        reason = f"device {device} is not supported by the batch-invariant kernels"
    else:
        reason = "the shape is outside the batch-invariant kernels' contract"
    logger.info(
        "Deterministic mode: %s keeps its previous NVFP4 kernel: %s.", what, reason
    )


# FlashInfer's batch-invariant NVFP4 quantizer takes bf16 / fp16 / fp32 rows of at
# most this many bytes with K a multiple of the block size.
_BI_QUANT_MAX_ROW_BYTES = 131072
_BI_QUANT_BLOCK = 16
_BI_QUANT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def nvfp4_batch_invariant_eligible(x: torch.Tensor) -> bool:
    """Whether ``x``'s NVFP4 activation quantization runs FlashInfer's batch-invariant
    kernel: the mode's predicate on its device and the kernel's row contract (dtype,
    ``K % 16``, row bytes); outside it the site keeps its current kernel."""
    k = x.shape[-1]
    if (
        use_flashinfer_bi_nvfp4(x.device)
        and x.dtype in _BI_QUANT_DTYPES
        and k % _BI_QUANT_BLOCK == 0
        and k * x.element_size() <= _BI_QUANT_MAX_ROW_BYTES
    ):
        return True
    _log_nvfp4_fallback_once(f"NVFP4 quantization of K = {k} ({x.dtype})", x.device)
    return False


def fp4_gemm_batch_invariant_eligible(device: torch.device, n: int, k: int) -> bool:
    """Whether a dense NVFP4 weight ``(n, k)`` runs ``mm_fp4(batch_invariant=True)``
    under the deterministic mode. Read at weight preparation and at every GEMM
    call from the same inputs, so the weight layout and the kernel agree."""
    if use_flashinfer_bi_nvfp4(device) and k % 32 == 0 and n % 8 == 0:
        return True
    _log_nvfp4_fallback_once(f"the dense NVFP4 GEMM (N = {n}, K = {k})", device)
    return False


fp4_quantize = None
try:
    from flashinfer import SfLayout as _SfLayout
    from flashinfer import fp4_quantize as _flashinfer_fp4_quantize
    from flashinfer import nvfp4_quantize as _flashinfer_nvfp4_quantize

    _flashinfer_fp4_quantize_backend = "cute-dsl" if get_platform().is_sm100 else "cuda"

    def _round_up(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    def _flashinfer_fp4_quantize_impl(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
        batch_invariant: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_invariant:
            # FlashInfer selects its batch-invariant kernel under backend="auto";
            # the scale layout is the caller's.
            return _flashinfer_fp4_quantize(
                input=input,
                global_scale=global_scale,
                sf_vec_size=sf_vec_size,
                sf_use_ue8m0=sf_use_ue8m0,
                is_sf_swizzled_layout=is_sf_swizzled_layout,
                is_sf_8x4_layout=is_sf_8x4_layout,
                enable_pdl=enable_pdl,
                backend="auto",
                batch_invariant=True,
            )
        return _flashinfer_fp4_quantize(
            input=input,
            global_scale=global_scale,
            sf_vec_size=sf_vec_size,
            sf_use_ue8m0=sf_use_ue8m0,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            is_sf_8x4_layout=is_sf_8x4_layout,
            enable_pdl=enable_pdl,
            backend=_flashinfer_fp4_quantize_backend,
        )

    def _flashinfer_fp4_quantize_fake(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
        batch_invariant: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_column_major = input.stride(-2) == 1
        if is_column_major:
            m = input.shape[-1]
            K = input.shape[-2]
        else:
            m = input.numel() // input.shape[-1]
            K = input.shape[-1]
        if is_column_major:
            x_q = input.new_empty((*input.shape[:-2], K // 2, m), dtype=torch.uint8)
        else:
            x_q = input.new_empty((*input.shape[:-1], K // 2), dtype=torch.uint8)
        if is_sf_swizzled_layout:
            row_size = 8 if is_sf_8x4_layout else 128
            sf_rows = _round_up(m, row_size)
            sf_cols = _round_up(K // sf_vec_size, 4)
        else:
            sf_rows = m
            sf_cols = K // sf_vec_size
        if is_column_major:
            sf = input.new_empty((sf_cols, sf_rows), dtype=torch.uint8)
        else:
            sf = input.new_empty((sf_rows, sf_cols), dtype=torch.uint8)
        return x_q, sf

    fp4_quantize = register_custom_op_from_extern(
        _flashinfer_fp4_quantize_impl,
        op_name="flashinfer_fp4_quantize",
        fake_impl=_flashinfer_fp4_quantize_fake,
    )
except ImportError:
    fp4_quantize = None


def nvfp4_quantize_per_token_linear(
    x: torch.Tensor, global_scale_inv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashInfer's per-token NVFP4 activation quantizer with linear block
    scales: the batch-invariant kernel under the deterministic mode, the
    CuTe-DSL kernel otherwise. Returns ``(fp4, block_scales, per_token_scale)``."""
    if nvfp4_batch_invariant_eligible(x):
        return _flashinfer_nvfp4_quantize(
            x,
            global_scale_inv,
            sfLayout=_SfLayout.layout_linear,
            per_token_activation=True,
            backend="auto",
            batch_invariant=True,
        )
    return _flashinfer_nvfp4_quantize(
        x,
        global_scale_inv,
        sfLayout=_SfLayout.layout_linear,
        per_token_activation=True,
        backend="cute-dsl",
    )


class Fp4GemmRunnerBackend(Enum):
    """Enum for FP4 GEMM runner backend selection."""

    AUTO = "auto"
    FLASHINFER_CUDNN = "flashinfer_cudnn"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    MARLIN = "marlin"

    def is_auto(self) -> bool:
        return self == Fp4GemmRunnerBackend.AUTO

    def is_flashinfer_cudnn(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUDNN

    def is_flashinfer_cutlass(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_trtllm(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_TRTLLM

    def is_flashinfer_cutedsl(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL

    def is_marlin(self) -> bool:
        return self == Fp4GemmRunnerBackend.MARLIN

    def is_flashinfer(self) -> bool:
        return self.value.startswith("flashinfer_")

    def supports_swiglu_fusion(self) -> bool:
        """Whether weight preparation builds the fused SwiGLU layout."""
        return not (self.is_marlin() or self.is_flashinfer_trtllm())

    def get_flashinfer_backend(self) -> str:
        """Get the backend string to pass to FlashInfer's mm_fp4 API.

        This remaps SGLang's user-facing backend names to FlashInfer's API names.
        Examples:
            'flashinfer_trtllm' -> 'trtllm'
            'flashinfer_cutlass' -> 'cutlass'
            'flashinfer_cudnn' -> 'cudnn'
            'flashinfer_cutedsl' -> 'cute-dsl'
        """
        if self == Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL:
            return "cute-dsl"
        if self.value.startswith("flashinfer_"):
            return self.value.removeprefix("flashinfer_")
        else:
            return self.value


FP4_GEMM_RUNNER_BACKEND: Fp4GemmRunnerBackend | None = None


def initialize_fp4_gemm_config() -> None:
    """Initialize the FP4 GEMM backend from the published configuration."""
    global FP4_GEMM_RUNNER_BACKEND

    backend = get_exec().kernel.fp4_gemm_runner_backend
    if backend == "auto":
        if get_platform().is_sm100:
            backend = "flashinfer_cutedsl"
        elif is_cuda() and (10, 0) > get_device_capability() >= (8, 0):
            backend = "marlin"
        else:
            backend = "flashinfer_cutlass"

    FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend(backend)


def get_fp4_gemm_runner_backend() -> Fp4GemmRunnerBackend:
    """Get the current FP4 GEMM runner backend."""
    global FP4_GEMM_RUNNER_BACKEND
    if FP4_GEMM_RUNNER_BACKEND is None:
        FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend.AUTO
    return FP4_GEMM_RUNNER_BACKEND
