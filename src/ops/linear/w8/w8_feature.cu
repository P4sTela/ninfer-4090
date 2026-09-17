#include "ops/linear/w8/w8_feature.h"

#include "core/device.h"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <array>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <int ActiveCols>
void launch_small(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = W8DFlash2FeatureProjectionGeometry;
    constexpr int kTileTokens = ActiveCols <= 8    ? 8
                                : ActiveCols <= 16 ? 16
                                : ActiveCols <= 24 ? 24
                                : ActiveCols <= 32 ? 32
                                : ActiveCols <= 40 ? 40
                                                     : 48;
    using Schedule = W8SmallTMmaSchedule<
        (ActiveCols <= 32 ? 8 : 4), kTileTokens, 2,
        (ActiveCols > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct)>;
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule, W8ContiguousOutput>
        <<<Geometry::kOutputRows / 16, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

using SmallLaunch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <std::size_t... I>
constexpr auto small_launchers(std::index_sequence<I...>) {
    return std::array<SmallLaunch, sizeof...(I)>{&launch_small<static_cast<int>(I) + 1>...};
}

constexpr auto kSmallLaunchers = small_launchers(std::make_index_sequence<48>{});

template <int Rows>
void launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    // Keep the predicated profile even on full tiles: the Full specialization regresses T=64.
    // A single activation stage makes K128 fit while retaining the common code/scale pipeline.
    using Schedule = W8RowSplitMmaGemmSchedule<Rows, 64, 16, 16, 1, 2, 128, 1>;
    const dim3 grid(weight.n / Rows, (x.ne[1] + 63) / 64);
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    w8_rowsplit_gemm_mma_kernel<Schedule, false><<<grid, Schedule::THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, weight.n, weight.k, x.ne[1],
        weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_w8_feature_small_t(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > static_cast<std::int32_t>(kSmallLaunchers.size())) {
        throw std::invalid_argument("W8 DFlash2 feature projection small-T: unsupported T");
    }
    kSmallLaunchers[static_cast<std::size_t>(x.ne[1] - 1)](x, w, out, stream);
}

void launch_w8_feature_r16_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch<16>(x, w, out, stream);
}

void launch_w8_feature_r32_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch<32>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
