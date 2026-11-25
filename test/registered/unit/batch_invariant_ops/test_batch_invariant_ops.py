# Adapted from https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/test_batch_invariance.py
import math
import unittest

import torch

from sglang.srt.batch_invariant_ops import batch_invariant_ops
from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Note: MI300 (gfx942) has 64KB shared memory limit but kernel needs 66KB
# MI35x (gfx950/CDNA4) may have different limits - testing on MI35x only
register_cuda_ci(est_time=20, stage="nightly", runner_config="1-gpu-large")
# B200 runs MXFP8 and NVFP4 quantization and GEMMs through FlashInfer's batch-invariant kernels.
register_cuda_ci(est_time=60, stage="nightly", runner_config="4-gpu-b200")
register_amd_ci(est_time=10, suite="nightly-test-1-gpu-amd-mi35x", nightly=True)

device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)

# Just to get the logging out of the way
with set_batch_invariant_mode(True):
    pass


def _cuda_kernel_names(fn):
    """The CUDA kernel names one call of ``fn`` launches (torch.profiler)."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return [
        e.name for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]


class TestBatchInvariantOps(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        batch_invariant_ops._ENABLE_MM_COMPARISON_TEST = True

    @classmethod
    def tearDownClass(cls):
        batch_invariant_ops._ENABLE_MM_COMPARISON_TEST = False

    def _test_batch_invariance(self, M, K, N, dtype):
        """
        Test that matrix operations produce identical results for:
        - Method 1: Matrix-vector multiplication (batch size 1)
        - Method 2: Matrix-matrix multiplication, then slice (full batch)
        """
        a = torch.linspace(-100, 100, M * K, dtype=dtype).reshape(M, K)

        # Create non-contiguous tensor
        b = torch.linspace(-100, 100, K * N, dtype=dtype).reshape(N, K)
        b = b.transpose(0, 1)

        # Method 1: Matrix-vector multiplication (batch size 1)
        out1 = torch.mm(a[:1], b)

        # Method 2: Matrix-matrix multiplication, then slice (full batch)
        out2_pre = torch.mm(a, b)
        out2 = out2_pre[:1]

        # Check if results are identical
        diff = (out1 - out2).abs().max()
        return diff.item()

    def _run_multiple_iterations(self, iters, M, K, N, dtype):
        """Run multiple iterations and collect diff statistics"""
        difflist = []
        for _ in range(iters):
            diff = self._test_batch_invariance(M, K, N, dtype)
            difflist.append(diff)
        return difflist

    def _assert_batch_invariant_results(self, difflist, dtype, test_name):
        """
        Assert that in batch-invariant mode:
        1. All diffs must not be NaN
        2. All diffs must be exactly 0
        3. Max, min, and diff of diffs must all be 0
        """
        max_diff = max(difflist)
        min_diff = min(difflist)
        diff_range = max_diff - min_diff

        # Check for NaN values
        self.assertFalse(
            math.isnan(max_diff), f"{test_name}: max_diff is NaN for {dtype}"
        )
        self.assertFalse(
            math.isnan(min_diff), f"{test_name}: min_diff is NaN for {dtype}"
        )
        self.assertFalse(
            math.isnan(diff_range), f"{test_name}: diff_range is NaN for {dtype}"
        )

        # Check that all diffs are exactly 0
        self.assertEqual(
            max_diff,
            0.0,
            f"{test_name}: max_diff must be 0 in batch-invariant mode, got {max_diff} for {dtype}",
        )
        self.assertEqual(
            min_diff,
            0.0,
            f"{test_name}: min_diff must be 0 in batch-invariant mode, got {min_diff} for {dtype}",
        )
        self.assertEqual(
            diff_range,
            0.0,
            f"{test_name}: diff_range must be 0 in batch-invariant mode, got {diff_range} for {dtype}",
        )

    def test_deepgemm_random_bf16_batch_invariance(self):
        if not batch_invariant_ops.ENABLE_JIT_DEEPGEMM:
            self.skipTest("DeepGEMM is unavailable on this device")

        deep_gemm = batch_invariant_ops.deep_gemm
        get_deterministic = getattr(deep_gemm, "get_deterministic_algorithms", None)
        original_mode = get_deterministic() if get_deterministic else None
        modes = (False, True) if get_deterministic else (None,)
        generator = torch.Generator(device=device_type).manual_seed(42)
        a = torch.randn(257, 4096, dtype=torch.bfloat16, generator=generator)
        b = torch.randn(4096, 4096, dtype=torch.bfloat16, generator=generator).T
        try:
            for mode in modes:
                if mode is not None:
                    deep_gemm.use_deterministic_algorithms(mode)
                with self.subTest(deterministic=mode):
                    ref = batch_invariant_ops._matmul_persistent_deepgemm(
                        a[:1], b, out_dtype=torch.bfloat16
                    )
                    for batch_size in (16, 64, 257):
                        out = batch_invariant_ops._matmul_persistent_deepgemm(
                            a[:batch_size], b, out_dtype=torch.bfloat16
                        )
                        torch.testing.assert_close(out[:1], ref, rtol=0, atol=0)
                    if get_deterministic:
                        self.assertEqual(get_deterministic(), mode)
        finally:
            if original_mode is not None:
                deep_gemm.use_deterministic_algorithms(original_mode)

    def test_flashinfer_mxfp8_batch_invariance(self):
        """On SM100-family parts the deterministic mode quantizes MXFP8
        activations and runs the dense MXFP8 GEMM through FlashInfer's
        batch-invariant kernels: the wrapper gives their bytes, a row's bits do
        not depend on the batch it is in, and the dense linear launches exactly
        those two kernels (the bits alone cannot tell the routes apart on parts
        where the previous kernels are invariant too)."""
        from sglang.srt.layers.quantization import fp8_utils
        from sglang.srt.layers.quantization.batch_invariant_quant import (
            flashinfer_bi_quant_supported,
            use_flashinfer_bi_quantizers,
        )

        if not flashinfer_bi_quant_supported(torch.cuda.current_device()):
            self.skipTest("FlashInfer batch-invariant MXFP8 kernels are unavailable")
        from flashinfer import SfLayout, mm_mxfp8, mxfp8_quantize

        generator = torch.Generator(device=device_type).manual_seed(0)
        K, N = 4096, 256
        x = torch.randn(257, K, dtype=torch.bfloat16, generator=generator)
        with set_batch_invariant_mode(True):
            flag = use_flashinfer_bi_quantizers(x.device)
            self.assertTrue(flag)
            for swizzled in (True, False):
                with self.subTest(swizzled=swizzled):
                    q, sf = fp8_utils.flashinfer_mxfp8_quantize(
                        x, swizzled, 32, batch_invariant=flag
                    )
                    q_ref, sf_ref = mxfp8_quantize(
                        x,
                        is_sf_swizzled_layout=swizzled,
                        alignment=32,
                        backend="auto",
                        sf_swizzle_layout=(
                            SfLayout.layout_128x4
                            if swizzled
                            else SfLayout.layout_linear
                        ),
                        batch_invariant=True,
                    )
                    torch.testing.assert_close(
                        q.view(torch.uint8), q_ref.view(torch.uint8), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        sf.view(torch.uint8), sf_ref.view(torch.uint8), rtol=0, atol=0
                    )
            # Row 0 alone against row 0 inside a batch; the scales are read in the
            # linear layout (the swizzled layout places the same bytes).
            q1, sf1 = fp8_utils.flashinfer_mxfp8_quantize(
                x[:1], False, 32, batch_invariant=flag
            )
            for m in (16, 64, 257):
                qm, sfm = fp8_utils.flashinfer_mxfp8_quantize(
                    x[:m], False, 32, batch_invariant=flag
                )
                torch.testing.assert_close(
                    qm[:1].view(torch.uint8), q1.view(torch.uint8), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    sfm.view(torch.uint8).view(m, -1)[0],
                    sf1.view(torch.uint8).view(1, -1)[0],
                    rtol=0,
                    atol=0,
                )
            # The dense linear: the batch-invariant quantizer, then the GEMM.
            weight = torch.randn(N, K, dtype=torch.bfloat16, generator=generator)
            w_q, w_sf = mxfp8_quantize(weight, is_sf_swizzled_layout=True)
            outs = {
                m: fp8_utils.flashinfer_mxfp8_blockscaled_linear(
                    x[:m], w_q, w_sf, backend="cute-dsl"
                )
                for m in (1, 64, 257)
            }
            for m in (64, 257):
                torch.testing.assert_close(outs[m][:1], outs[1], rtol=0, atol=0)
            for m in (1, 64, 257):
                names = _cuda_kernel_names(
                    lambda: fp8_utils.flashinfer_mxfp8_blockscaled_linear(
                        x[:m], w_q, w_sf, backend="cute-dsl"
                    )
                )
                bi_kernels = [k for k in names if "batch_invariant" in k]
                self.assertEqual(len(bi_kernels), 2, names)
                self.assertEqual(len(set(bi_kernels)), 2, names)
            x_q, x_sf = mxfp8_quantize(
                x, is_sf_swizzled_layout=True, backend="auto", batch_invariant=True
            )
            expected = mm_mxfp8(
                x_q,
                w_q.t(),
                x_sf,
                w_sf,
                out_dtype=torch.bfloat16,
                backend="auto",
                batch_invariant=True,
            )
            torch.testing.assert_close(outs[257], expected, rtol=0, atol=0)

    def test_flashinfer_mxfp8_fallbacks(self):
        """Under the mode, an input outside FlashInfer's batch-invariant kernels
        (rows over 128 KiB) and the env opt-out run the previous kernels: the
        output is the pinned path's bit for bit and no batch-invariant kernel is
        launched (CUPTI kernel names)."""
        from sglang.srt.environ import envs
        from sglang.srt.layers.quantization import fp8_utils
        from sglang.srt.layers.quantization.batch_invariant_quant import (
            flashinfer_bi_quant_supported,
        )

        if not flashinfer_bi_quant_supported(torch.cuda.current_device()):
            self.skipTest("FlashInfer batch-invariant MXFP8 kernels are unavailable")
        from flashinfer import mxfp8_quantize

        kernels = _cuda_kernel_names
        generator = torch.Generator(device=device_type).manual_seed(0)
        lin = fp8_utils.flashinfer_mxfp8_blockscaled_linear
        cases = {
            # rows over 128 KiB: the quantizer and the GEMM keep the previous kernels
            "long rows": (
                torch.randn(16, 66048, dtype=torch.bfloat16, generator=generator),
                4096,
                False,
            ),
            # the env opt-out keeps both previous kernels
            "opt-out": (
                torch.randn(16, 4096, dtype=torch.bfloat16, generator=generator),
                256,
                True,
            ),
        }
        for name, (x, n, opt_out) in cases.items():
            with self.subTest(case=name):
                weight = torch.randn(
                    n, x.shape[1], dtype=torch.bfloat16, generator=generator
                )
                w_q, w_sf = mxfp8_quantize(weight, is_sf_swizzled_layout=True)
                with set_batch_invariant_mode(True):
                    with (
                        envs.SGLANG_ENABLE_DETERMINISTIC_FLASHINFER_BI_KERNELS.override(
                            False
                        )
                    ):
                        pinned = lin(x, w_q, w_sf, backend="cute-dsl", pin_tactic=True)
                    if opt_out:
                        with envs.SGLANG_ENABLE_DETERMINISTIC_FLASHINFER_BI_KERNELS.override(
                            False
                        ):
                            out = lin(x, w_q, w_sf, backend="cute-dsl")
                            names = kernels(
                                lambda: lin(x, w_q, w_sf, backend="cute-dsl")
                            )
                    else:
                        out = lin(x, w_q, w_sf, backend="cute-dsl")
                        names = kernels(lambda: lin(x, w_q, w_sf, backend="cute-dsl"))
                torch.testing.assert_close(out, pinned, rtol=0, atol=0)
                bi_kernels = [k for k in names if "batch_invariant" in k]
                self.assertFalse(bi_kernels, bi_kernels)

    def test_flashinfer_nvfp4_batch_invariance(self):
        """On SM100-family parts the deterministic mode runs NVFP4 activation
        quantization and the dense NVFP4 GEMM on FlashInfer's batch-invariant
        kernels: the wrapper's bytes are the direct call's, and a row's codes,
        scales and GEMM output do not depend on the rows around it."""
        from sglang.srt.layers.quantization import fp4_utils
        from sglang.srt.layers.quantization.modelopt_quant import fp4_gemm

        device = torch.device("cuda", torch.cuda.current_device())
        with set_batch_invariant_mode(True):
            if not fp4_utils.use_flashinfer_bi_nvfp4(device):
                self.skipTest("FlashInfer batch-invariant NVFP4 is unavailable")
            import flashinfer

            bi = fp4_utils.use_flashinfer_bi_nvfp4(device)
            generator = torch.Generator(device=device_type).manual_seed(0)
            K, N = 4096, 2048
            x = torch.randn(257, K, dtype=torch.bfloat16, generator=generator)
            gs = (448.0 * 6.0 / x.float().abs().max()).reshape(1)
            gs_inv = torch.full((1,), 1.0 / (448.0 * 6.0), dtype=torch.float32)
            same = lambda a, b: torch.testing.assert_close(a, b, rtol=0, atol=0)
            u8 = lambda t: t.view(torch.uint8)
            # The wrapper under the mode: the bytes of the direct FlashInfer call,
            # dense 128x4 and MoE linear scale layouts, and the per-token entry.
            for swizzled in (True, False):
                q, sf = fp4_utils.fp4_quantize(
                    x, gs, is_sf_swizzled_layout=swizzled, batch_invariant=bi
                )
                q_ref, sf_ref = flashinfer.fp4_quantize(
                    x,
                    gs,
                    is_sf_swizzled_layout=swizzled,
                    backend="auto",
                    batch_invariant=True,
                )
                same(q, q_ref)
                same(u8(sf), u8(sf_ref))
            q, sf, pts = fp4_utils.nvfp4_quantize_per_token_linear(x, gs_inv)
            q_ref, sf_ref, pts_ref = flashinfer.nvfp4_quantize(
                x,
                gs_inv,
                sfLayout=flashinfer.SfLayout.layout_linear,
                per_token_activation=True,
                backend="auto",
                batch_invariant=True,
            )
            same(q, q_ref), same(u8(sf), u8(sf_ref)), same(pts, pts_ref)
            # Row 0 alone against row 0 inside a batch: codes in both layouts, the
            # row's scales in the linear (row-addressed) layout, the per-token scale.
            q1, sf1 = fp4_utils.fp4_quantize(
                x[:1], gs, is_sf_swizzled_layout=False, batch_invariant=bi
            )
            q1s, _ = fp4_utils.fp4_quantize(x[:1], gs, batch_invariant=bi)
            p1, psf1, pts1 = fp4_utils.nvfp4_quantize_per_token_linear(x[:1], gs_inv)
            for m in (16, 64, 257):
                q, sf = fp4_utils.fp4_quantize(
                    x[:m], gs, is_sf_swizzled_layout=False, batch_invariant=bi
                )
                qs, _ = fp4_utils.fp4_quantize(x[:m], gs, batch_invariant=bi)
                p, psf, pts = fp4_utils.nvfp4_quantize_per_token_linear(x[:m], gs_inv)
                same(q[:1], q1), same(u8(sf)[:1], u8(sf1)), same(qs[:1], q1s)
                same(p[:1], p1), same(u8(psf)[:1], u8(psf1)), same(pts[:1], pts1)
            # The dense GEMM path, the weight in the 128x4 layout the loader
            # prepares under the mode.
            w = torch.randn(N, K, dtype=torch.bfloat16, generator=generator)
            gw = (448.0 * 6.0 / w.float().abs().max()).reshape(1)
            w_q, w_sf = fp4_utils.fp4_quantize(w, gw, batch_invariant=bi)
            alpha = (1.0 / (gs * gw)).to(torch.float32)

            def dense(rows):
                x_q, x_sf = fp4_utils.fp4_quantize(rows, gs, batch_invariant=bi)
                return fp4_gemm(x_q, w_q.T, x_sf, w_sf.T, alpha, torch.bfloat16, N)

            out1 = dense(x[:1])

            for m in (64, 257):
                same(dense(x[:m])[:1], out1)

            # The bit comparison is blind where the other kernels happen to be

            # invariant too: the path must launch the batch-invariant quantizer and

            # GEMM, two distinct kernels, at every M.

            for m in (1, 64, 257):
                names = _cuda_kernel_names(lambda m=m: dense(x[:m]))

                bi_kernels = [k for k in names if "batch_invariant" in k]

                self.assertEqual(len(bi_kernels), 2, names)

                self.assertEqual(len(set(bi_kernels)), 2, names)

    def test_small_matrices(self):
        """Test batch invariance with small matrix sizes"""
        test_cases = [
            ("Small-1", 8, 64, 128),
            ("Small-2", 16, 128, 256),
            ("Small-3", 4, 32, 64),
        ]

        for name, M, K, N in test_cases:
            with self.subTest(name=name, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_multiple_iterations(
                                iters=5, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_medium_matrices(self):
        """Test batch invariance with medium matrix sizes"""
        test_cases = [
            ("Medium-1", 32, 128, 1024),
            ("Medium-2", 64, 512, 2048),
            ("Medium-3", 24, 192, 768),
        ]

        for name, M, K, N in test_cases:
            with self.subTest(name=name, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_multiple_iterations(
                                iters=5, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_large_matrices(self):
        """Test batch invariance with large matrix sizes"""
        test_cases = [
            ("Large-1", 128, 1024, 4096),
            ("Large-2", 256, 2048, 8192),
            ("Large-3", 96, 768, 3072),
        ]

        for name, M, K, N in test_cases:
            with self.subTest(name=name, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_multiple_iterations(
                                iters=5, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def _test_bmm_batch_invariance(self, B, M, K, N, dtype):
        """
        Test that BMM operations produce identical results for:
        - Method 1: BMM with subset of batches
        - Method 2: BMM with all batches, then slice
        """
        a = torch.linspace(-100, 100, B * M * K, dtype=dtype).reshape(B, M, K)
        b = torch.linspace(-100, 100, B * K * N, dtype=dtype).reshape(B, K, N)

        # Method 1: BMM with subset (first 2 batches)
        subset_size = min(2, B)
        out1 = torch.bmm(a[:subset_size], b[:subset_size])

        # Method 2: BMM with all batches, then slice
        out2_pre = torch.bmm(a, b)
        out2 = out2_pre[:subset_size]

        # Check if results are identical
        diff = (out1 - out2).abs().max()
        return diff.item()

    def _run_bmm_multiple_iterations(self, iters, B, M, K, N, dtype):
        """Run multiple BMM iterations and collect diff statistics"""
        difflist = []
        for _ in range(iters):
            diff = self._test_bmm_batch_invariance(B, M, K, N, dtype)
            difflist.append(diff)
        return difflist

    def test_bmm_small_matrices(self):
        """Test BMM batch invariance with small matrix sizes"""
        test_cases = [
            ("BMM-Small-1", 4, 8, 64, 128),
            ("BMM-Small-2", 8, 16, 128, 256),
            ("BMM-Small-3", 6, 4, 32, 64),
        ]

        for name, B, M, K, N in test_cases:
            with self.subTest(name=name, B=B, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_bmm_multiple_iterations(
                                iters=5, B=B, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_bmm_medium_matrices(self):
        """Test BMM batch invariance with medium matrix sizes"""
        test_cases = [
            ("BMM-Medium-1", 8, 32, 128, 1024),
            ("BMM-Medium-2", 16, 64, 512, 2048),
            ("BMM-Medium-3", 12, 24, 192, 768),
        ]

        for name, B, M, K, N in test_cases:
            with self.subTest(name=name, B=B, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_bmm_multiple_iterations(
                                iters=5, B=B, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)

    def test_bmm_large_matrices(self):
        """Test BMM batch invariance with large matrix sizes"""
        test_cases = [
            ("BMM-Large-1", 16, 128, 1024, 4096),
            ("BMM-Large-2", 32, 256, 2048, 8192),
            ("BMM-Large-3", 24, 96, 768, 3072),
        ]

        for name, B, M, K, N in test_cases:
            with self.subTest(name=name, B=B, M=M, K=K, N=N):
                for dtype in [torch.float32, torch.bfloat16]:
                    with self.subTest(dtype=dtype):
                        # Run with batch-invariant mode
                        with set_batch_invariant_mode(True):
                            difflist = self._run_bmm_multiple_iterations(
                                iters=5, B=B, M=M, K=K, N=N, dtype=dtype
                            )
                            self._assert_batch_invariant_results(difflist, dtype, name)


if __name__ == "__main__":
    unittest.main()
