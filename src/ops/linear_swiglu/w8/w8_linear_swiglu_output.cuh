#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Each CTA computes eight gate/up row pairs. The second half of the schedule's sixteen-row
// staging tile addresses the corresponding SwiGLU rows in the upper half of the projection.
template <int Intermediate>
struct W8SwiGluPairedRows {
    static constexpr int kOutputRowsPerCta = 8;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & (kOutputRowsPerCta - 1)) +
               (local_row >= kOutputRowsPerCta ? Intermediate : 0);
    }
};

struct W8SwiGluDirectEpilogue {
    __nv_bfloat16* out;
    int rows;

    template <int ActiveCols>
    __device__ __forceinline__ void store_pair(int row, int col0, float4 projected) const {
        if (col0 < ActiveCols) {
            out[static_cast<std::int64_t>(col0) * rows + row] =
                __float2bfloat16_rn(silu(projected.x) * projected.z);
        }
        if (col0 + 1 < ActiveCols) {
            out[static_cast<std::int64_t>(col0 + 1) * rows + row] =
                __float2bfloat16_rn(silu(projected.y) * projected.w);
        }
    }
};

} // namespace ninfer::ops::detail
