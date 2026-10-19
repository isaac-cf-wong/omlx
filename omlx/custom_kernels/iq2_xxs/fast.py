# SPDX-License-Identifier: Apache-2.0
"""iq2_xxs quantized matmul kernels.

``iq2_xxs`` is GGML's IQ2_XXS format (2.0625 bits per weight) with each
block's fp16 ``d`` split out of the packed words, so a quantized linear layer
stores

* ``weight``  uint32  ``[..., N, K / 16]``
* ``scales``  float16 ``[..., N, K / 256]``

and computes ``y = x @ W.T``. ``K`` must be a multiple of 256; the block count
per row may be odd. The bit layout is documented in
``csrc/iq2_xxs_common.h``.

Kernels are JIT-compiled through ``mx.fast.metal_kernel`` from the same Metal
sources the native extension builds, so the two paths share one definition of
the format.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import mlx.core as mx

_CSRC = Path(__file__).parent / "csrc"

# Rows of x up to which the matvec kernel beats dequantize + matmul. Above
# it the weights are dequantized once and the dense matmul reuses them.
QMV_MAX_ROWS = 8

_NR = 4  # output rows per simdgroup
_NSG = 2  # simdgroups per threadgroup


def _read_header() -> str:
    text = []
    for name in ("iq2_xxs_tables.h", "iq2_xxs_common.h"):
        for line in (_CSRC / name).read_text().splitlines():
            if line.startswith(("#pragma once", "#include")):
                continue
            text.append(line)
    return "\n".join(text)


@cache
def _header() -> str:
    return _read_header()


_DEQUANT_SOURCE = """
    constexpr uint NB32 = K / 32;
    const uint sb = thread_position_in_grid.x;
    const ulong r = thread_position_in_grid.y;
    if (sb >= NB32) {
        return;
    }
    const device uint* q = w + r * (K / 16) + 2 * sb;
    const uint a0 = q[0];
    const uint a1 = q[1];
    const float db =
        omlx_iq2xxs_db(float(scales[r * (K / 256) + sb / 8]), a1) * 0.25f;
    device T* o = out + r * K + 32 * sb;
    for (short l = 0; l < 4; ++l) {
        constant uchar* g =
            (constant uchar*)(omlx_iq2xxs_grid + ((a0 >> (8 * l)) & 0xFFu));
        const uint s = omlx_iq2xs_signs((a1 >> (7 * l)) & 127u);
        for (short k = 0; k < 8; ++k) {
            const float v = db * float(g[k]);
            o[8 * l + k] = T(((s >> k) & 1u) ? -v : v);
        }
    }
"""

# One threadgroup computes NR * NSG output columns for one row of x. Each
# lane walks the row's 32-weight sub-blocks with stride 32, so any block
# count (including odd ones) is covered. {INDEXING} picks the x row and the
# expert for output row a (see _qmv_kernel).
_QMV_SOURCE = """
    constexpr int NB32 = K / 32;
    constexpr ulong K16 = K / 16;
    constexpr ulong K256 = K / 256;

    const uint a = threadgroup_position_in_grid.y;
    {INDEXING}
    const int first_row =
        (threadgroup_position_in_grid.x * NSG + simdgroup_index_in_threadgroup) * NR;

    const device T* xr = x + xrow * K;
    const device uint* wb = w + e * N * K16;
    const device half* sbase = scales + e * N * K256;

    float sumf[NR];
    for (int r = 0; r < NR; ++r) {
        sumf[r] = 0.0f;
    }

    for (int ib = thread_index_in_simdgroup; ib < NB32; ib += 32) {
        float yl[32];
        const device vec<T, 4>* x4 = (const device vec<T, 4>*)(xr + 32 * ib);
        for (short i = 0; i < 8; ++i) {
            const float4 v = float4(x4[i]);
            yl[4 * i] = v.x;
            yl[4 * i + 1] = v.y;
            yl[4 * i + 2] = v.z;
            yl[4 * i + 3] = v.w;
        }
        for (int r = 0; r < NR; ++r) {
            const int row = first_row + r;
            if (row >= N) {
                break;
            }
            const uint2 q = *((const device uint2*)(wb + ulong(row) * K16 + 2 * ib));
            const float db = omlx_iq2xxs_db(float(sbase[ulong(row) * K256 + ib / 8]), q.y);
            sumf[r] += db * omlx_iq2xxs_dot(yl, q.x, q.y);
        }
    }

    for (int r = 0; r < NR; ++r) {
        const float total = simd_sum(sumf[r]);
        const int row = first_row + r;
        if (thread_index_in_simdgroup == 0 && row < N) {
            out[ulong(a) * N + row] = T(omlx_iq2xxs_act<ACT>(total * 0.25f));
        }
    }
"""

# Row selection per kernel kind: a dense matrix, a gather whose x row is
# a / XDIV (mlx-lm's SwitchGLU layout, no index arrays to build), and a
# gather with explicit x-row indices.
_QMV_INDEXING = {
    "dense": "const ulong xrow = a; const ulong e = 0;",
    "gather_div": "const ulong xrow = a / XDIV; const ulong e = rhs_indices[a];",
    "gather": "const ulong xrow = lhs_indices[a]; const ulong e = rhs_indices[a];",
}
_QMV_INPUTS = {
    "dense": ["x", "w", "scales"],
    "gather_div": ["x", "w", "scales", "rhs_indices"],
    "gather": ["x", "w", "scales", "lhs_indices", "rhs_indices"],
}


# Gathered matmul for many rows (MoE prefill). Assignments arrive sorted by
# expert, so consecutive rows mostly share a weight matrix. A threadgroup
# owns a BM x BN output tile; within it, each run of rows with the same
# expert is computed separately: stage BM x 32 of x and dequantize BN x 32
# of that expert's weights (one IQ2 sub-block per row) into threadgroup
# memory, then multiply with 8x8 simdgroup matrices. Four simdgroups split
# the tile 2 x 2, each owning 16 x 32 of it.
_QMM_SOURCE = """
    constexpr int BM = 32;
    constexpr int BN = 64;
    constexpr int BK = 32;
    constexpr int NTHREADS = 128;
    constexpr int NB32 = K / 32;
    constexpr ulong K16 = K / 16;
    constexpr ulong K256 = K / 256;

    threadgroup float xs[BM * BK];
    threadgroup float ws[BN * BK];
    // The output tile is staged in ws once a run's K loop is done (same size).
    threadgroup float* cs = ws;
    constant uint2* grid = (constant uint2*)omlx_iq2xxs_grid;

    const uint tid = thread_index_in_threadgroup;

    const uint n_rows = lhs_indices_shape[0];
    const uint m0 = threadgroup_position_in_grid.y * BM;
    const uint n0 = threadgroup_position_in_grid.x * BN;
    const uint mend = min(m0 + BM, n_rows);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint sg_m = sg / 2;
    const uint sg_n = sg % 2;

    // Per-thread staging coordinates.
    const uint xr = tid / 4;
    const uint xc = (tid % 4) * 8;
    const uint wr = tid / 2;
    const uint wl = (tid % 2) * 2;

    uint run_start = m0;
    while (run_start < mend) {
        const uint e = rhs_indices[run_start];
        uint run_end = run_start + 1;
        while (run_end < mend && rhs_indices[run_end] == e) {
            run_end++;
        }

        simdgroup_matrix<float, 8, 8> acc[2][4];
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }

        const device uint* wb = w + ulong(e) * N * K16;
        const device half* sbase = scales + ulong(e) * N * K256;
        const uint xa = m0 + xr;
        const bool x_valid = xa >= run_start && xa < run_end;
        const device T* xp = x + ulong(x_valid ? lhs_indices[xa] : 0) * K + xc;
        const uint wn = n0 + wr;

        for (int kb = 0; kb < NB32; ++kb) {
            for (int j = 0; j < 8; ++j) {
                xs[xr * BK + xc + j] = x_valid ? float(xp[kb * 32 + j]) : 0.0f;
            }
            if (wn < N) {
                const uint2 q = *((const device uint2*)(wb + ulong(wn) * K16 + 2 * kb));
                const uint a0 = q.x;
                const uint a1 = q.y;
                const float db =
                    omlx_iq2xxs_db(float(sbase[ulong(wn) * K256 + kb / 8]), a1) * 0.25f;
                for (uint l = wl; l < wl + 2; ++l) {
                    const uint2 g = grid[(a0 >> (8 * l)) & 0xFFu];
                    const uint s = omlx_iq2xs_signs((a1 >> (7 * l)) & 127u);
                    for (uint k = 0; k < 4; ++k) {
                        const float v0 = db * float((g.x >> (8 * k)) & 0xFFu);
                        const float v1 = db * float((g.y >> (8 * k)) & 0xFFu);
                        ws[wr * BK + 8 * l + k] = (s & (1u << k)) ? -v0 : v0;
                        ws[wr * BK + 8 * l + 4 + k] = (s & (1u << (k + 4))) ? -v1 : v1;
                    }
                }
            } else {
                for (uint k = 0; k < 16; ++k) {
                    ws[wr * BK + 8 * wl + k] = 0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int kk = 0; kk < BK; kk += 8) {
                simdgroup_matrix<float, 8, 8> af[2];
                simdgroup_matrix<float, 8, 8> bf[4];
                for (int i = 0; i < 2; ++i) {
                    simdgroup_load(af[i], xs + (sg_m * 16 + 8 * i) * BK + kk, BK);
                }
                for (int j = 0; j < 4; ++j) {
                    simdgroup_load(
                        bf[j], ws + (sg_n * 32 + 8 * j) * BK + kk, BK, ulong2(0, 0), true);
                }
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 4; ++j) {
                simdgroup_store(
                    acc[i][j], cs + (sg_m * 16 + 8 * i) * BN + sg_n * 32 + 8 * j, BN);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < BM * BN; idx += NTHREADS) {
            const uint a = m0 + idx / BN;
            const uint n = n0 + idx % BN;
            if (a >= run_start && a < run_end && n < N) {
                out[ulong(a) * N + n] = T(omlx_iq2xxs_act<ACT>(cs[idx]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        run_start = run_end;
    }
"""

_QMM_BM = 32
_QMM_BN = 64

# Assignments at or above which the gathered matmul sorts by expert and uses
# the tiled kernel; below it every assignment is an independent matvec.
GATHER_QMM_MIN_ROWS = 64


_ACTIVATIONS = {None: 0, "relu2": 1}


def _act_id(activation: str | None) -> int:
    try:
        return _ACTIVATIONS[activation]
    except KeyError:
        raise ValueError(f"unsupported fused activation {activation!r}") from None


@cache
def _gather_qmm_kernel():
    return mx.fast.metal_kernel(
        name="omlx_iq2xxs_gather_qmm",
        input_names=["x", "w", "scales", "lhs_indices", "rhs_indices"],
        output_names=["out"],
        source=_QMM_SOURCE,
        header=_header(),
    )


def gather_qmm_sorted(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    lhs_indices: mx.array,
    rhs_indices: mx.array,
    activation: str | None = None,
) -> mx.array:
    """``gather_qmv`` semantics for ``rhs_indices`` sorted ascending.

    Rows sharing an expert must be contiguous; the result is still correct
    when they are not, only slower.
    """
    if w.ndim != 3:
        raise ValueError(
            f"gather_qmm expects stacked [E, N, K/16] weights, got {w.shape}"
        )
    n, k = _check(w, scales)
    if x.ndim != 2 or x.shape[-1] != k:
        raise ValueError(f"x must be [R, {k}], got {x.shape}")
    lhs = lhs_indices.astype(mx.uint32).reshape(-1)
    rhs = rhs_indices.astype(mx.uint32).reshape(-1)
    rows = lhs.size
    return _gather_qmm_kernel()(
        inputs=[x, w, scales, lhs, rhs],
        template=[("T", x.dtype), ("K", k), ("N", n), ("ACT", _act_id(activation))],
        grid=(-(-n // _QMM_BN) * 128, -(-rows // _QMM_BM), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _dequant_kernel():
    return mx.fast.metal_kernel(
        name="omlx_iq2xxs_dequantize",
        input_names=["w", "scales"],
        output_names=["out"],
        source=_DEQUANT_SOURCE,
        header=_header(),
    )


@cache
def _qmv_kernel(kind: str):
    return mx.fast.metal_kernel(
        name=f"omlx_iq2xxs_qmv_{kind}",
        input_names=_QMV_INPUTS[kind],
        output_names=["out"],
        source=_QMV_SOURCE.replace("{INDEXING}", _QMV_INDEXING[kind]),
        header=_header(),
    )


def _check(w: mx.array, scales: mx.array) -> tuple[int, int]:
    if w.dtype != mx.uint32:
        raise TypeError(f"iq2_xxs weight must be uint32, got {w.dtype}")
    if scales.dtype != mx.float16:
        raise TypeError(f"iq2_xxs scales must be float16, got {scales.dtype}")
    n, k16 = w.shape[-2], w.shape[-1]
    k = k16 * 16
    if k == 0 or k % 256 != 0:
        raise ValueError(
            f"iq2_xxs row length must be a positive multiple of 256, got {k}"
        )
    if scales.shape[:-1] != w.shape[:-1] or scales.shape[-1] != k // 256:
        raise ValueError(
            f"iq2_xxs scales shape {scales.shape} does not match weight shape {w.shape}"
        )
    return n, k


def dequantize(w: mx.array, scales: mx.array, dtype: mx.Dtype = mx.float16) -> mx.array:
    """Return the dense ``[..., N, K]`` weights."""
    n, k = _check(w, scales)
    rows = w.size // (k // 16)
    out = _dequant_kernel()(
        inputs=[w, scales],
        template=[("T", dtype), ("K", k)],
        grid=(k // 32, rows, 1),
        threadgroup=(min(k // 32, 256), 1, 1),
        output_shapes=[(*w.shape[:-1], k)],
        output_dtypes=[dtype],
    )[0]
    return out


def _launch_qmv(kind, x, w, scales, n, k, rows, extra=(), xdiv=1, activation=None):
    groups = -(-n // (_NR * _NSG))
    template = [
        ("T", x.dtype),
        ("K", k),
        ("N", n),
        ("NR", _NR),
        ("NSG", _NSG),
        ("ACT", _act_id(activation)),
    ]
    if kind == "gather_div":
        template.append(("XDIV", xdiv))
    return _qmv_kernel(kind)(
        inputs=[x, w, scales, *extra],
        template=template,
        grid=(groups * _NSG * 32, rows, 1),
        threadgroup=(_NSG * 32, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )[0]


def qmv(x: mx.array, w: mx.array, scales: mx.array) -> mx.array:
    """``x @ W.T`` for a single (non-stacked) weight, any number of rows."""
    if w.ndim != 2:
        raise ValueError(f"qmv expects a 2-D weight, got shape {w.shape}")
    n, k = _check(w, scales)
    if x.shape[-1] != k:
        raise ValueError(f"x has {x.shape[-1]} features, weight expects {k}")
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    out = _launch_qmv("dense", x2, w, scales, n, k, x2.shape[0])
    return out.reshape(*lead, n)


def gather_qmv(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    lhs_indices: mx.array,
    rhs_indices: mx.array,
    activation: str | None = None,
) -> mx.array:
    """Row ``a`` of the result is ``x[lhs_indices[a]] @ w[rhs_indices[a]].T``.

    ``x`` is ``[R, K]``, ``w`` is ``[E, N, K/16]``, both index arrays are 1-D
    of equal length ``A``; the result is ``[A, N]``.
    """
    if w.ndim != 3:
        raise ValueError(
            f"gather_qmv expects stacked [E, N, K/16] weights, got {w.shape}"
        )
    n, k = _check(w, scales)
    if x.ndim != 2 or x.shape[-1] != k:
        raise ValueError(f"x must be [R, {k}], got {x.shape}")
    lhs = lhs_indices.astype(mx.uint32).reshape(-1)
    rhs = rhs_indices.astype(mx.uint32).reshape(-1)
    if lhs.size != rhs.size:
        raise ValueError("lhs_indices and rhs_indices must have the same length")
    return _launch_qmv(
        "gather", x, w, scales, n, k, lhs.size, (lhs, rhs), activation=activation
    )


def iq2_xxs_matmul(x: mx.array, w: mx.array, scales: mx.array) -> mx.array:
    """``x @ W.T`` choosing matvec or dequantize + matmul by row count."""
    rows = x.size // x.shape[-1]
    if rows <= QMV_MAX_ROWS:
        return qmv(x, w, scales)
    return x @ dequantize(w, scales, x.dtype).T


def _gather_rows(x2, w, scales, n, k, lhs, rhs, activation):
    """Gathered rows: matvec per row when few, else sort by expert and tile."""
    if rhs.size < GATHER_QMM_MIN_ROWS:
        return gather_qmv(x2, w, scales, lhs, rhs, activation)
    order = mx.argsort(rhs)
    out = gather_qmm_sorted(x2, w, scales, lhs[order], rhs[order], activation)
    return out[mx.argsort(order)]


def gather_iq2_xxs_matmul(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    rhs_indices: mx.array,
    lhs_indices: mx.array | None = None,
    activation: str | None = None,
) -> mx.array:
    """Gathered ``x @ W[e].T`` with ``mx.gather_qmm`` broadcasting semantics.

    ``x`` is ``[..., M, K]`` and broadcasts against ``rhs_indices``'s shape
    ``[...]`` the way ``mx.gather_qmm`` does with ``transpose=True``; the
    result is ``[..., M, N]``. ``activation`` ("relu2") is applied to the
    output inside the kernel.
    """
    n, k = _check(w, scales)
    m = x.shape[-2]
    x_lead = x.shape[:-2]
    idx_shape = rhs_indices.shape

    # mlx-lm switch layers: x is [..., 1, 1, K] (one row per token, shared by
    # its k experts) or [..., k, 1, K] (one row per assignment).
    if lhs_indices is None and m == 1 and idx_shape:
        if x_lead == idx_shape:
            xdiv = 1
        elif x_lead == (*idx_shape[:-1], 1):
            xdiv = idx_shape[-1]
        else:
            xdiv = 0
        if xdiv:
            rhs = rhs_indices.astype(mx.uint32).reshape(-1)
            x2 = x.reshape(-1, k)
            if rhs.size < GATHER_QMM_MIN_ROWS:
                out = _launch_qmv(
                    "gather_div",
                    x2,
                    w,
                    scales,
                    n,
                    k,
                    rhs.size,
                    (rhs,),
                    xdiv=xdiv,
                    activation=activation,
                )
            else:
                lhs = mx.arange(rhs.size, dtype=mx.uint32) // xdiv
                out = _gather_rows(x2, w, scales, n, k, lhs, rhs, activation)
            return out.reshape(*idx_shape, 1, n)

    if lhs_indices is None:
        src = mx.arange(x.size // (m * k), dtype=mx.uint32).reshape(x_lead)
        batch = mx.broadcast_shapes(x_lead, idx_shape)
    else:
        src = lhs_indices.astype(mx.uint32)
        batch = mx.broadcast_shapes(src.shape, idx_shape)
    # Map every (batch, m) output row to its source row of x instead of
    # materializing the broadcast x.
    src = mx.broadcast_to(src, batch).reshape(-1, 1)
    lhs = (src * m + mx.arange(m, dtype=mx.uint32)).reshape(-1)
    rhs = mx.broadcast_to(rhs_indices.astype(mx.uint32), batch).reshape(-1, 1)
    rhs = mx.broadcast_to(rhs, (rhs.shape[0], m)).reshape(-1)
    out = _gather_rows(x.reshape(-1, k), w, scales, n, k, lhs, rhs, activation)
    return out.reshape(*batch, m, n)
