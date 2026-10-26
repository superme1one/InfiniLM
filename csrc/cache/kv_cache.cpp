#include "kv_cache.hpp"

#include "../utils.hpp"
#include "infinicore/ops.hpp"
#include <dlfcn.h>
#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

#ifndef INFINILM_V_CACHE_DECODE_LAYOUT_PATCH
#define INFINILM_V_CACHE_DECODE_LAYOUT_PATCH 1
#endif
#ifndef INFINILM_V_CACHE_DECODE_LAYOUT_SINGLE_STORAGE
#define INFINILM_V_CACHE_DECODE_LAYOUT_SINGLE_STORAGE 1
#endif
#ifndef INFINILM_V_CACHE_DECODE_LAYOUT_INPLACE_TRANSPOSE
#define INFINILM_V_CACHE_DECODE_LAYOUT_INPLACE_TRANSPOSE 1
#endif
#ifndef INFINILM_V_CACHE_DECODE_LAYOUT_PADDED_STORAGE
#define INFINILM_V_CACHE_DECODE_LAYOUT_PADDED_STORAGE 1
#endif

#ifndef INFINILM_V_CACHE_DECODE_LAYOUT_LAYER_FALLBACK
#define INFINILM_V_CACHE_DECODE_LAYOUT_LAYER_FALLBACK 1
#endif

bool infiniLmVCacheDecodeLayoutLayerFallsBack(size_t layer_idx) {
    const char *spec =
        std::getenv("INFINILM_V_CACHE_DECODE_LAYOUT_FALLBACK_LAYERS");
    if (spec == nullptr || spec[0] == '\0') {
        return false;
    }

    const char *cursor = spec;
    while (*cursor != '\0') {
        while (*cursor == ' ' || *cursor == ',') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }

        char *start_end = nullptr;
        const size_t start = static_cast<size_t>(
            std::strtoull(cursor, &start_end, 10));
        if (start_end == cursor) {
            while (*cursor != '\0' && *cursor != ',') {
                ++cursor;
            }
            continue;
        }

        size_t stop = start;
        cursor = start_end;
        if (*cursor == '-') {
            ++cursor;
            char *stop_end = nullptr;
            const size_t parsed_stop = static_cast<size_t>(
                std::strtoull(cursor, &stop_end, 10));
            if (stop_end != cursor) {
                stop = parsed_stop;
                cursor = stop_end;
            }
        }
        size_t lower = start;
        size_t upper = stop;
        if (upper < lower) {
            const size_t swap = lower;
            lower = upper;
            upper = swap;
        }
        if (layer_idx >= lower && layer_idx <= upper) {
            return true;
        }

        while (*cursor != '\0' && *cursor != ',') {
            ++cursor;
        }
    }
    return false;
}

bool infiniLmVCacheDecodeLayoutModeSupported(const char *mode) {
    return mode != nullptr &&
        (std::strcmp(mode, "page16_fp32_forward") == 0 ||
         std::strcmp(mode, "page16_fp32_warp") == 0 ||
         std::strcmp(mode, "page32_fp32_forward") == 0 ||
         std::strcmp(mode, "page16_fp32_reverse") == 0 ||
         std::strcmp(mode, "page32_fp32_reverse") == 0 ||
         std::strcmp(mode, "page16_bf16_forward") == 0 ||
         std::strcmp(mode, "page32_bf16_forward") == 0);
}

const char *infiniLmVCacheDecodeLayoutMode() {
    const char *mode = std::getenv("INFINILM_V_CACHE_DECODE_LAYOUT");
    return infiniLmVCacheDecodeLayoutModeSupported(mode) ? mode : nullptr;
}

size_t infiniLmVCacheDecodeLayoutPageSize() {
    const char *mode = infiniLmVCacheDecodeLayoutMode();
    return mode != nullptr && std::strncmp(mode, "page32_", 7) == 0 ? 32 : 16;
}

using InfiniLmVCacheDecodeLayoutUpdateFn = int (*)(
    void *,
    const void *,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    ptrdiff_t,
    ptrdiff_t,
    ptrdiff_t,
    ptrdiff_t);

InfiniLmVCacheDecodeLayoutUpdateFn
infiniLmResolveVCacheDecodeLayoutUpdate() {
    constexpr const char *symbol = "infiniopVCacheDecodeLayoutUpdateBf16";
    static auto fn = reinterpret_cast<InfiniLmVCacheDecodeLayoutUpdateFn>(
        dlsym(RTLD_DEFAULT, symbol));
    if (fn == nullptr) {
        if (auto *handle = dlopen("libinfiniop.so", RTLD_LAZY | RTLD_GLOBAL)) {
            fn = reinterpret_cast<InfiniLmVCacheDecodeLayoutUpdateFn>(
                dlsym(handle, symbol));
        }
    }
    return fn;
}

using InfiniLmVCacheDecodeLayoutTransposeFn = int (*)(
    void *,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t);

InfiniLmVCacheDecodeLayoutTransposeFn
infiniLmResolveVCacheDecodeLayoutTranspose() {
    constexpr const char *symbol =
        "infiniopVCacheDecodeLayoutTransposeInPlaceBf16";
    static auto fn = reinterpret_cast<InfiniLmVCacheDecodeLayoutTransposeFn>(
        dlsym(RTLD_DEFAULT, symbol));
    if (fn == nullptr) {
        if (auto *handle = dlopen("libinfiniop.so", RTLD_LAZY | RTLD_GLOBAL)) {
            fn = reinterpret_cast<InfiniLmVCacheDecodeLayoutTransposeFn>(
                dlsym(handle, symbol));
        }
    }
    return fn;
}

size_t infiniLmKvPadAlign() {
    const char *env = std::getenv("INFINILM_KV_PAD_ALIGN");
    if (env == nullptr || env[0] == '\0') {
        return 0;
    }
    try {
        return static_cast<size_t>(std::stoull(env));
    } catch (...) {
        return 0;
    }
}

size_t alignUp(size_t value, size_t align) {
    return align == 0 ? value : ((value + align - 1) / align) * align;
}

bool infiniLmKvPadDebug() {
    const char *env = std::getenv("INFINILM_KV_PAD_DEBUG");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}
} // namespace

namespace infinilm::cache {
// ==========================
// StaticKVCacheConfig
// ==========================

StaticKVCacheConfig::StaticKVCacheConfig(
    infinicore::Size _max_batch_size,
    infinicore::Size _max_cache_len)
    : max_batch_size_(_max_batch_size),
      max_cache_len_(_max_cache_len) {
}

std::unique_ptr<CacheConfig>
StaticKVCacheConfig::unique_copy() const {
    return std::make_unique<StaticKVCacheConfig>(*this);
}

infinicore::Size
StaticKVCacheConfig::max_batch_size() const {
    return max_batch_size_;
}

infinicore::Size
StaticKVCacheConfig::max_cache_len() const {
    return max_cache_len_;
}

// ==========================
// StaticKVCache
// ==========================

StaticKVCache::StaticKVCache(
    infinicore::Size k_dim,
    infinicore::Size v_dim,
    infinicore::Size num_k_heads,
    infinicore::Size num_v_heads,
    infinicore::Size num_layers,
    infinicore::Size max_positional_embedding,
    infinicore::DataType dtype,
    const StaticKVCacheConfig &config,
    const engine::distributed::RankInfo &rank_info)
    : Cache(),
      k_dim_(k_dim),
      v_dim_(v_dim),
      num_rank_k_heads_(num_k_heads / rank_info.tp_size),
      num_rank_v_heads_(num_v_heads / rank_info.tp_size),
      rank_batch_size_(config.max_batch_size()),
      cache_len_(config.max_cache_len() == std::numeric_limits<infinicore::Size>::max() || config.max_cache_len() == 0 ? max_positional_embedding : config.max_cache_len()),
      rank_num_layers_(num_layers),
      dtype_(dtype) {

    const char *decode_layout_mode =
        infiniLmVCacheDecodeLayoutMode();
    v_decode_layout_storage_len_ = cache_len_;
    if (decode_layout_mode != nullptr &&
        dtype_ == infinicore::DataType::BF16) {
        v_decode_layout_enabled_ = true;
        v_decode_layout_page_size_ = infiniLmVCacheDecodeLayoutPageSize();
        v_decode_layout_storage_len_ =
            ((cache_len_ + v_decode_layout_page_size_ - 1) /
             v_decode_layout_page_size_) *
            v_decode_layout_page_size_;
        v_decode_layout_num_pages_ =
            v_decode_layout_storage_len_ /
            v_decode_layout_page_size_;
        v_decode_layout_ready_.assign(rank_num_layers_, false);
    }

    // Allocate K/V cache. When KV padding experiments are enabled, use zeros so
    // unread padded rows are deterministic.
    if (infiniLmKvPadAlign() > 1) {
        k_caches_ = infinicore::Tensor::zeros(
            {rank_num_layers_,
             rank_batch_size_,
             num_rank_k_heads_,
             cache_len_,
             k_dim_},
            dtype_,
            rank_info.device);
        v_caches_ = infinicore::Tensor::zeros(
            {rank_num_layers_,
             rank_batch_size_,
             num_rank_v_heads_,
             v_decode_layout_storage_len_,
             v_dim_},
            dtype_,
            rank_info.device);
    } else {
        k_caches_ = infinicore::Tensor::empty(
            {rank_num_layers_,
             rank_batch_size_,
             num_rank_k_heads_,
             cache_len_,
             k_dim_},
            dtype_,
            rank_info.device);
        v_caches_ = infinicore::Tensor::empty(
            {rank_num_layers_,
             rank_batch_size_,
             num_rank_v_heads_,
             v_decode_layout_storage_len_,
             v_dim_},
            dtype_,
            rank_info.device);
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
StaticKVCache::update(size_t layer_idx,
                      const infinicore::Tensor &k,
                      const infinicore::Tensor &v,
                      const infinicore::Tensor &past_sequence_lengths) {
    ASSERT(layer_idx < rank_num_layers_);

    auto batch_size = k->size(0);
    auto update_len = k->size(2);
    size_t cache_pos = reinterpret_cast<int64_t *>(past_sequence_lengths->to(infinicore::Device::cpu())->data())[0];
    auto result_len = cache_pos + update_len;

    ASSERT(result_len <= cache_len_);

    ASSERT_EQ(batch_size, rank_batch_size_);

    auto k_cache_layer = k_caches_->narrow({{0, layer_idx, 1}})->squeeze(0);
    auto v_cache_layer = v_caches_->narrow({{0, layer_idx, 1}})->squeeze(0);

    auto k_cache_update = k_cache_layer->narrow({{2, cache_pos, update_len}});
    k_cache_update->copy_from(k);

    if (v_decode_layout_enabled_ &&
        !infiniLmVCacheDecodeLayoutLayerFallsBack(layer_idx)) {
        if (!v_decode_layout_ready_[layer_idx] &&
            update_len == 1 && cache_pos > 0) {
            auto transpose_fn =
                infiniLmResolveVCacheDecodeLayoutTranspose();
            if (transpose_fn == nullptr) {
                throw std::runtime_error(
                    "V-cache in-place transpose symbol is not available");
            }
            const int transpose_status = transpose_fn(
                v_cache_layer->data(),
                batch_size,
                num_rank_v_heads_,
                cache_pos,
                v_dim_,
                v_decode_layout_storage_len_,
                v_decode_layout_page_size_);
            if (transpose_status != 0) {
                throw std::runtime_error(
                    "V-cache in-place transpose failed");
            }
            v_decode_layout_ready_[layer_idx] = true;
        }
        if (v_decode_layout_ready_[layer_idx]) {
            auto update_fn = infiniLmResolveVCacheDecodeLayoutUpdate();
            if (update_fn == nullptr) {
                throw std::runtime_error(
                    "V-cache decode-layout update symbol is not available");
            }
            const int status = update_fn(
                v_cache_layer->data(),
                v->data(),
                batch_size,
                num_rank_v_heads_,
                update_len,
                v_dim_,
                cache_pos,
                v_decode_layout_num_pages_,
                v_decode_layout_page_size_,
                v->stride(0),
                v->stride(1),
                v->stride(2),
                v->stride(3));
            if (status != 0) {
                throw std::runtime_error(
                    "V-cache decode-layout update failed");
            }
        } else {
            v_cache_layer
                ->narrow({{2, cache_pos, update_len}})
                ->copy_from(v);
        }
    } else {
        v_cache_layer
            ->narrow({{2, cache_pos, update_len}})
            ->copy_from(v);
    }

    size_t return_len = result_len;
    size_t pad_align = infiniLmKvPadAlign();
    if (pad_align > 1 && update_len == 1) {
        size_t padded_len = alignUp(result_len, pad_align);
        if (padded_len <= cache_len_) {
            return_len = padded_len;
        }
    }
    if (infiniLmKvPadDebug() && return_len != result_len && layer_idx == 0) {
        std::cerr << "INFINILM_KV_PAD real_len=" << result_len
                  << " padded_len=" << return_len
                  << " align=" << pad_align << std::endl;
    }

    auto k_total = k_cache_layer->narrow({{2, 0, return_len}});
    // Before the first decode step V remains token-major for prefill. Once a
    // layer has been transposed in place, decode reads it through the layout
    // getter and this placeholder tensor is not dereferenced.
    auto v_total =
        v_decode_layout_enabled_ && v_decode_layout_ready_[layer_idx]
        ? v
        : v_cache_layer->narrow({{2, 0, return_len}});

    return {k_total, v_total};
}

bool
StaticKVCache::has_v_decode_layout(size_t layer_idx) const {
    return v_decode_layout_enabled_ &&
        layer_idx < v_decode_layout_ready_.size() &&
        v_decode_layout_ready_[layer_idx];
}

size_t StaticKVCache::v_decode_layout_page_size() const {
    return v_decode_layout_page_size_;
}

infinicore::Tensor
StaticKVCache::get_v_decode_layout(size_t layer_idx) const {
    ASSERT(layer_idx < rank_num_layers_);
    if (!has_v_decode_layout(layer_idx)) {
        throw std::runtime_error("V-cache decode layout is not enabled");
    }
    auto layer = v_caches_->narrow({{0, layer_idx, 1}})->squeeze(0);
    return layer->view(
        {rank_batch_size_,
         num_rank_v_heads_,
         v_decode_layout_num_pages_,
         v_dim_,
         v_decode_layout_page_size_});
}

// ==========================
// PagedKVCacheConfig
// ==========================
PagedKVCacheConfig::PagedKVCacheConfig(
    size_t num_blocks,
    size_t block_size)
    : num_blocks_(num_blocks),
      block_size_(block_size) {
}

std::unique_ptr<CacheConfig>
PagedKVCacheConfig::unique_copy() const {
    return std::make_unique<PagedKVCacheConfig>(*this);
}

size_t
PagedKVCacheConfig::num_blocks() const {
    return num_blocks_;
}

size_t
PagedKVCacheConfig::block_size() const {
    return block_size_;
}

// ==========================
// PagedKVCache
// ==========================
PagedKVCache::PagedKVCache(
    infinicore::Size k_dim,
    infinicore::Size v_dim,
    infinicore::Size num_k_heads,
    infinicore::Size num_v_heads,
    infinicore::Size num_layers,
    infinicore::DataType dtype,
    const PagedKVCacheConfig &config,
    const engine::distributed::RankInfo &rank_info)
    : Cache(),
      k_dim_(k_dim),
      v_dim_(v_dim),
      num_rank_k_heads_(num_k_heads / rank_info.tp_size),
      num_rank_v_heads_(num_v_heads / rank_info.tp_size),
      rank_num_layers_(num_layers),
      dtype_(dtype),
      num_blocks_per_layer_(config.num_blocks()),
      block_size_(config.block_size()) {
    // [num_layers, num_blocks, num_rank_k_heads, block_size, k_dim]
    k_caches_ = infinicore::Tensor::empty(
        {rank_num_layers_,
         num_blocks_per_layer_,
         num_rank_k_heads_,
         block_size_,
         k_dim_},
        dtype_,
        rank_info.device);

    // [num_layers, num_blocks, num_rank_v_heads, block_size, v_dim]
    v_caches_ = infinicore::Tensor::empty(
        {rank_num_layers_,
         num_blocks_per_layer_,
         num_rank_v_heads_,
         block_size_,
         v_dim_},
        dtype_,
        rank_info.device);
}

std::tuple<infinicore::Tensor, infinicore::Tensor> PagedKVCache::update(
    size_t layer_idx,
    const infinicore::Tensor &k,
    const infinicore::Tensor &v,
    const infinicore::Tensor &slot_mapping) {

    auto &&[k_cache_layer, v_cache_layer] = get_paged_kv(layer_idx);

    infinicore::op::paged_caching_(
        k_cache_layer,
        v_cache_layer,
        k,
        v,
        slot_mapping);
    return {k_cache_layer, v_cache_layer};
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
PagedKVCache::get_paged_kv(size_t layer_idx) {
    auto k_cache_layer = k_caches_->narrow({{0, layer_idx, 1}})->squeeze(0);
    auto v_cache_layer = v_caches_->narrow({{0, layer_idx, 1}})->squeeze(0);
    return {k_cache_layer, v_cache_layer};
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
PagedKVCache::get_contiguous_kv(
    size_t layer_idx,
    const infinicore::Tensor block_tables,
    const infinicore::Tensor cache_lens,
    const infinicore::Tensor input_offsets,
    size_t request_id) {
    ASSERT_EQ(block_tables->dtype(), infinicore::DataType::I64);
    ASSERT_EQ(cache_lens->dtype(), infinicore::DataType::I64);
    ASSERT_EQ(input_offsets->dtype(), infinicore::DataType::I64);

    auto nreq = block_tables->size(0);
    auto block_tables_cpu = block_tables->to(infinicore::Device::cpu());
    auto cache_lens_cpu = cache_lens->to(infinicore::Device::cpu());
    auto input_offsets_cpu = input_offsets->to(infinicore::Device::cpu());
    infinicore::context::syncDevice();

    // [num_blocks, num_rank_v_heads, block_size, v_dim]
    auto &&[k_cache_layer, v_cache_layer] = get_paged_kv(layer_idx);

    auto req = request_id;
    auto cache_lens_ptr = reinterpret_cast<const int64_t *>(cache_lens_cpu->data());
    auto input_offsets_ptr = reinterpret_cast<const int64_t *>(input_offsets_cpu->data());
    int64_t total_len = cache_lens_ptr[req] + (input_offsets_ptr[req + 1] - input_offsets_ptr[req]);

    auto full_k = infinicore::Tensor::empty(
        {num_rank_k_heads_, (size_t)total_len, k_dim_},
        k_cache_layer->dtype(), k_cache_layer->device());

    auto full_v = infinicore::Tensor::empty(
        {num_rank_v_heads_, (size_t)total_len, v_dim_},
        v_cache_layer->dtype(), v_cache_layer->device());

    size_t nblocks = total_len / block_size_;
    size_t r = total_len % block_size_;

    for (size_t b = 0; b < nblocks; b++) {
        size_t bid = *((int64_t *)(block_tables_cpu->narrow({{0, req, 1}, {1, b, 1}})->data()));

        full_k->narrow({{1, b * block_size_, block_size_}})
            ->copy_from(k_cache_layer->narrow({{0, bid, 1}})->squeeze(0));
        full_v->narrow({{1, b * block_size_, block_size_}})
            ->copy_from(v_cache_layer->narrow({{0, bid, 1}})->squeeze(0));
    }

    if (r > 0) {
        size_t bid = *((int64_t *)(block_tables_cpu->narrow({{0, req, 1}, {1, nblocks, 1}})->data()));

        full_k->narrow({{1, nblocks * block_size_, r}})
            ->copy_from(k_cache_layer->narrow({{0, bid, 1}})->squeeze(0)->narrow({{1, 0, r}}));
        full_v->narrow({{1, nblocks * block_size_, r}})
            ->copy_from(v_cache_layer->narrow({{0, bid, 1}})->squeeze(0)->narrow({{1, 0, r}}));
    }

    return {full_k, full_v};
}

} // namespace infinilm::cache
