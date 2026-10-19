// SPDX-License-Identifier: Apache-2.0
//
// Device helpers for the iq2_xxs quantization mode.
//
// Layout (MLX split of GGML block_iq2_xxs, one row of K weights):
//   weight  uint32 [..., N, K/16]   two words per 32-weight sub-block
//   scales  half   [..., N, K/256]  the block's d
// For sub-block b of a 256-weight block j, a0 = weight[16j + 2b] holds four
// codebook indices (one per byte) and a1 = weight[16j + 2b + 1] holds four
// 7-bit sign indices (bits 0..27) and a 4-bit sub-block scale (bits 28..31).
// Weight (8l + k) of the sub-block is
//   d * (0.5 + (a1 >> 28)) * 0.25 * grid[a0 byte l][k] * sign(a1, l, k).
// This matches GGML dequantize_row_iq2_xxs on the same bytes.

#pragma once

#include "iq2_xxs_tables.h"

// ksigns_iq2xs[i]: i with bit 7 set to the parity of i.
inline uint omlx_iq2xs_signs(uint i) {
  return i | ((popcount(i) & 1u) << 7);
}

// Sub-block scale without the trailing 0.25 (callers fold it in once).
inline float omlx_iq2xxs_db(float d, uint a1) {
  return d * (0.5f + float(a1 >> 28));
}

// Dot product of one 32-weight sub-block with yl[0..31], unscaled. Each
// codebook entry is read as one uint2 and split with 32-bit shifts: 64-bit
// shifts are emulated on Apple GPUs, and per-byte loads cost more.
inline float omlx_iq2xxs_dot(thread const float* yl, uint a0, uint a1) {
  constant uint2* grid = (constant uint2*)omlx_iq2xxs_grid;
  float sum = 0.0f;
  for (short l = 0; l < 4; ++l) {
    const uint2 g = grid[(a0 >> (8 * l)) & 0xFFu];
    const uint s = omlx_iq2xs_signs((a1 >> (7 * l)) & 127u);
    for (short k = 0; k < 4; ++k) {
      const float v0 = float((g.x >> (8 * k)) & 0xFFu);
      const float v1 = float((g.y >> (8 * k)) & 0xFFu);
      sum += yl[8 * l + k] * ((s & (1u << k)) ? -v0 : v0);
      sum += yl[8 * l + 4 + k] * ((s & (1u << (k + 4))) ? -v1 : v1);
    }
  }
  return sum;
}

// Optional activation fused into a kernel's output (ACT: 0 none, 1 relu^2).
template <int ACT>
inline float omlx_iq2xxs_act(float v) {
  if (ACT == 1) {
    v = max(v, 0.0f);
    return v * v;
  }
  return v;
}
