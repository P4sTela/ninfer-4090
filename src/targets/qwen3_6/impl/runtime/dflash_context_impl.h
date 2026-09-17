#include "targets/qwen3_6/impl/runtime/dflash_context.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

DFlashPersistentState::DFlashPersistentState(DeviceSpan backing,
                                             const DFlashPersistentLayout& layout,
                                             CyclicKVCache& local_state)
    : local(local_state),
      prefill_features(layout.prefill_features.bind(backing)),
      prefill_positions(layout.prefill_positions.bind(backing)),
      pending_features(layout.pending_features.bind(backing)) {
    if (local.layer_count() != DFlashConfig::local_layers ||
        local.capacity() != DFlashConfig::local_capacity ||
        local.num_kv_heads() != DFlashConfig::kv_heads ||
        local.head_dim() != DFlashConfig::head_dim) {
        throw std::invalid_argument("DFlash local cache layout is invalid");
    }
    if (layout.full) {
        full.emplace(backing, *layout.full);
        if (full->layers() != 1 || full->max_context() != layout.full->max_context ||
            full->page_pool().plane_count() != 2 ||
            full->page_pool().plane(0).dtype != DType::BF16 ||
            full->page_pool().plane(0).ne[0] != DFlashConfig::head_dim ||
            full->page_pool().plane(0).ne[1] != kPagedKVPageSize ||
            full->page_pool().plane(0).ne[3] != DFlashConfig::kv_heads) {
            throw std::invalid_argument("DFlash full cache layout is invalid");
        }
    }
}

CyclicKVCacheLayerView DFlashPersistentState::local_layer(std::uint32_t layer) const {
    return local.layer_view(layer);
}

PagedKVBatchLayerView DFlashPersistentState::full_batch_layer(std::uint32_t layer) const {
    if (!full) { throw std::logic_error("DFlash2 has no full KV cache"); }
    return full->batch_layer_view(layer);
}

void DFlashPersistentState::save_rewrite_checkpoint(std::int32_t source_slot,
                                                    std::int32_t destination_slot,
                                                    cudaStream_t stream) {
    local.copy_slot_from(local, source_slot, destination_slot, stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
