# SPDX-License-Identifier: Apache-2.0
"""Tests for the iq2_xxs (GGML IQ2_XXS, split scales) Metal kernels.

The reference decoder below follows the format description in
``omlx/custom_kernels/iq2_xxs/csrc/iq2_xxs_common.h`` in numpy; only the
codebook values are shared with the kernels, and those are pinned by a
hand-decoded block.
"""

from __future__ import annotations

import re

import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.iq2_xxs import fast

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="iq2_xxs kernels need Metal"
)


def _grid() -> np.ndarray:
    text = (fast._CSRC / "iq2_xxs_tables.h").read_text()
    vals = [int(v, 16) for v in re.findall(r"0x[0-9a-f]{16}", text)]
    assert len(vals) == 256
    return np.array(
        [[(v >> (8 * k)) & 0xFF for k in range(8)] for v in vals], dtype=np.float64
    )


def _ref_dequant(w: np.ndarray, scales: np.ndarray) -> np.ndarray:
    grid = _grid()
    *lead, n, k16 = w.shape
    w2 = w.reshape(-1, k16).astype(np.uint64)
    s2 = scales.reshape(-1, scales.shape[-1]).astype(np.float64)
    out = np.zeros((w2.shape[0], k16 * 16))
    for r in range(w2.shape[0]):
        for sb in range(k16 // 2):
            a0, a1 = int(w2[r, 2 * sb]), int(w2[r, 2 * sb + 1])
            db = s2[r, sb // 8] * (0.5 + (a1 >> 28)) * 0.25
            for sub in range(4):
                g = grid[(a0 >> (8 * sub)) & 0xFF]
                i = (a1 >> (7 * sub)) & 127
                signs = i | ((bin(i).count("1") & 1) << 7)
                for k in range(8):
                    sign = -1.0 if (signs >> k) & 1 else 1.0
                    out[r, 32 * sb + 8 * sub + k] = db * g[k] * sign
    return out.reshape(*lead, n, k16 * 16)


def _random_weights(shape_nk, seed=0, scale=1e-2):
    *lead, n, k = shape_nk
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2**32, size=(*lead, n, k // 16), dtype=np.uint64).astype(
        np.uint32
    )
    s = (rng.uniform(size=(*lead, n, k // 256)) * scale).astype(np.float16)
    return w, s


def _rel(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_codebook_matches_hand_decoded_block():
    # grid[0] is eight 8s; sign index 1 has parity 1, so bits 0 and 7 flip;
    # sub-block scale nibble 1 gives db = d * 1.5 * 0.25.
    w = np.zeros((1, 16), dtype=np.uint32)
    w[0, 1] = (1 << 28) | 1
    s = np.full((1, 1), 2.0, dtype=np.float16)
    out = np.array(fast.dequantize(mx.array(w), mx.array(s), mx.float32))[0]
    expected = np.full(256, 2.0 * 0.5 * 0.25 * 8)  # other sub-blocks: scale 0.5
    expected[:32] = 2.0 * 1.5 * 0.25 * 8
    expected[0] = expected[7] = -expected[0]
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("k", [256, 2304])
def test_dequantize_matches_reference(k):
    w, s = _random_weights((2, 6, k))
    out = np.array(fast.dequantize(mx.array(w), mx.array(s), mx.float32))
    np.testing.assert_allclose(out, _ref_dequant(w, s), rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("rows", [1, 3, 9])
@pytest.mark.parametrize("n,k", [(13, 2304), (40, 512)])
def test_qmv_matches_reference(rows, n, k):
    w, s = _random_weights((n, k), seed=rows)
    x = np.random.default_rng(1).standard_normal((rows, k)).astype(np.float32)
    ref = x.astype(np.float64) @ _ref_dequant(w, s).T
    out = fast.qmv(mx.array(x), mx.array(w), mx.array(s))
    assert _rel(out, ref) < 1e-5


def test_matmul_dequant_path_matches_reference():
    w, s = _random_weights((24, 512), seed=3)
    x = np.random.default_rng(2).standard_normal((fast.QMV_MAX_ROWS + 5, 512))
    x = x.astype(np.float32)
    ref = x.astype(np.float64) @ _ref_dequant(w, s).T
    assert _rel(fast.iq2_xxs_matmul(mx.array(x), mx.array(w), mx.array(s)), ref) < 1e-5


@pytest.mark.parametrize("tokens", [1, 3, 20])  # 9..180 assignments: qmv and qmm
@pytest.mark.parametrize("activation", [None, "relu2"])
def test_gather_switch_layouts_match_reference(tokens, activation):
    e, n, k, top_k = 5, 72, 512, 9
    w, s = _random_weights((e, n, k), seed=tokens)
    deq = _ref_dequant(w, s)
    rng = np.random.default_rng(tokens)
    idx = rng.integers(0, e, size=(tokens, top_k)).astype(np.uint32)
    x_tok = rng.standard_normal((tokens, 1, 1, k)).astype(np.float32)
    x_asg = rng.standard_normal((tokens, top_k, 1, k)).astype(np.float32)

    def act(v):
        return np.maximum(v, 0.0) ** 2 if activation == "relu2" else v

    ref_tok = act(np.einsum("tk,tjnk->tjn", x_tok[:, 0, 0], deq[idx]))
    ref_asg = act(np.einsum("tjk,tjnk->tjn", x_asg[:, :, 0], deq[idx]))
    for x, ref in ((x_tok, ref_tok), (x_asg, ref_asg)):
        out = fast.gather_iq2_xxs_matmul(
            mx.array(x), mx.array(w), mx.array(s), mx.array(idx), activation=activation
        )
        assert out.shape == (tokens, top_k, 1, n)
        assert _rel(np.array(out)[:, :, 0], ref) < 1e-5


def test_gather_qmm_sorted_handles_unsorted_and_partial_tiles():
    # 70 rows (not a multiple of the 32-row tile), 100 columns (not a
    # multiple of 64), experts deliberately unsorted.
    e, n, k, rows = 4, 100, 768, 70
    w, s = _random_weights((e, n, k), seed=7)
    deq = _ref_dequant(w, s)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((rows, k)).astype(np.float32)
    rhs = rng.integers(0, e, size=rows).astype(np.uint32)
    lhs = np.arange(rows, dtype=np.uint32)
    ref = np.einsum("ak,ank->an", x.astype(np.float64), deq[rhs])
    for order in (np.argsort(rhs, kind="stable"), np.arange(rows)):
        out = fast.gather_qmm_sorted(
            mx.array(x),
            mx.array(w),
            mx.array(s),
            mx.array(lhs[order]),
            mx.array(rhs[order]),
        )
        assert _rel(out, ref[order]) < 1e-5


def test_bfloat16_inputs_keep_dtype():
    w, s = _random_weights((16, 256), seed=4)
    x = mx.random.normal((2, 256)).astype(mx.bfloat16)
    assert fast.qmv(x, mx.array(w), mx.array(s)).dtype == mx.bfloat16


def test_rejects_bad_layouts():
    w, s = _random_weights((8, 256))
    with pytest.raises(TypeError):
        fast.qmv(mx.zeros((1, 256)), mx.array(w).astype(mx.int32), mx.array(s))
    with pytest.raises(TypeError):
        fast.qmv(mx.zeros((1, 256)), mx.array(w), mx.array(s).astype(mx.float32))
    with pytest.raises(ValueError):
        fast.qmv(
            mx.zeros((1, 128)),
            mx.zeros((8, 8), dtype=mx.uint32),
            mx.zeros((8, 0), mx.float16),
        )
    with pytest.raises(ValueError):
        fast.qmv(mx.zeros((1, 512)), mx.array(w), mx.array(s))
    with pytest.raises(ValueError):
        fast.gather_iq2_xxs_matmul(
            mx.zeros((1, 1, 1, 256)),
            mx.array(w)[None],
            mx.array(s)[None],
            mx.zeros((1, 1), mx.uint32),
            activation="gelu",
        )
