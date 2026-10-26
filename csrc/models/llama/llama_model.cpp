#include <infinicore/tensor.hpp>
#include <optional>
std::optional<infinicore::Tensor> iv_request_length_mirror(std::optional<infinicore::Tensor>,bool);
#include "llama_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "../profiler.hpp"
#include <iostream>

bool iv_chain_begin(bool);
void iv_chain_observe(infinicore::Tensor);
std::pair<infinicore::Tensor, infinicore::Tensor> iv_add_rms_fusion(
    infinicore::Tensor, infinicore::Tensor, infinicore::Tensor, float);

namespace infinilm::models::llama {

LlamaModel::LlamaModel(const LlamaConfig &config,
                       const infinicore::Device &device,
                       engine::distributed::RankInfo rank_info)
    : config_(config), rank_info_(rank_info) {
    const auto &dtype{config.dtype};
    // Initialize token embeddings
    INFINICORE_NN_MODULE_INIT(embed_tokens, config.vocab_size, config.hidden_size,
                              std::nullopt, dtype, device);

    // Initialize decoder layers with layer indices
    // TODO: Update INFINICORE_NN_MODULE_VEC_INIT macro to support per-layer constructor arguments
    //       (e.g., via a factory function or lambda that receives the layer index)
    //       Currently, we can't use the macro because each layer needs a different layer_idx
    layers_.reserve(config.num_hidden_layers);
    for (size_t i = 0; i < config.num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<LlamaDecoderLayer>(
            "layers." + std::to_string(i), config, device, i, rank_info));
    }

    // Initialize final layer normalization
    INFINICORE_NN_MODULE_INIT(norm, config.hidden_size, config.rms_norm_eps,
                              dtype, device);

    // Initialize Rotary Position Embeddings (shared across all layers)
    // Use GPT-J-style inverse frequencies (default) and GPT_NEOX rotation pairing
    INFINICORE_NN_MODULE_INIT(rotary_emb, config.head_dim, config.max_position_embeddings,
                              config.rope_theta, infinicore::nn::RoPE::Algo::GPT_NEOX,
                              dtype, device, config.rope_scaling);

    for (auto &layer : layers_) {
        if (layer) {
            layer->set_rotary_emb(rotary_emb_);
        }
    }
}

infinicore::Tensor LlamaModel::forward(const infinicore::Tensor &input_ids,
                                       const infinicore::Tensor &position_ids,
                                       std::optional<infinicore::Tensor> past_sequence_lengths,
                                       std::optional<infinicore::Tensor> total_sequence_lengths,
                                       std::optional<infinicore::Tensor> input_offsets,
                                       std::optional<infinicore::Tensor> block_tables,
                                       std::optional<infinicore::Tensor> slot_mapping) const {
    const auto phase = infinilm::profile::phase_from_tensor(input_ids);

    // Request-local mirror: static attention consumes CPU lengths; paged attention does not.
    total_sequence_lengths = iv_request_length_mirror(total_sequence_lengths,
        std::dynamic_pointer_cast<infinilm::cache::StaticKVCache>(kv_cache_) != nullptr);

    // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
    auto hidden_states = [&]() {
        INFINILM_PROFILE_SCOPE(phase, -1, rank_info_.tp_rank, "model_embed_tokens", input_ids);
        return embed_tokens_->forward(input_ids);
    }();

    const bool deferred = iv_chain_begin(rank_info_.tp_size == 1 && !layers_.empty() &&
        hidden_states->dtype() == infinicore::DataType::BF16 &&
        hidden_states->device().getType() == infinicore::Device::Type::ILUVATAR &&
        std::dynamic_pointer_cast<infinilm::cache::StaticKVCache>(kv_cache_) != nullptr);
    if (deferred) {
        std::optional<infinicore::Tensor> pending_residual;
        for (size_t i = 0; i < layers_.size(); ++i) {
            auto next = layers_.at(i)->forward_deferred(hidden_states, position_ids, kv_cache_,
                past_sequence_lengths, total_sequence_lengths, input_offsets, block_tables, slot_mapping,
                pending_residual);
            hidden_states = next.first;
            pending_residual = next.second;
        }
        auto output = iv_add_rms_fusion(pending_residual.value(), hidden_states,
            norm_->weight(), static_cast<float>(norm_->eps())).first;
        iv_chain_observe(output);
        return output;
    }

    // 2. Process through all decoder layers
    size_t num_layers = layers_.size();
    for (size_t i = 0; i < num_layers; ++i) {
        hidden_states = layers_.at(i)->forward(hidden_states, position_ids, kv_cache_, past_sequence_lengths, total_sequence_lengths, input_offsets, block_tables, slot_mapping);
    }

    auto final_output = [&]() {
        INFINILM_PROFILE_SCOPE(phase, -1, rank_info_.tp_rank, "model_final_norm", hidden_states);
        return norm_->forward(hidden_states);
    }();
    iv_chain_observe(final_output);
    return final_output;
}

void LlamaModel::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        kv_cache_ = nullptr;
        return;
    }
    if (auto kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config)) {
        kv_cache_ = std::make_shared<cache::StaticKVCache>(
            config_.head_dim,
            config_.head_dim,
            config_.num_key_value_heads,
            config_.num_key_value_heads,
            config_.num_hidden_layers,
            config_.max_position_embeddings,
            config_.dtype,
            *kv_cache_config,
            rank_info_);

    } else if (auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config)) {
        kv_cache_ = std::make_shared<cache::PagedKVCache>(
            config_.head_dim,
            config_.head_dim,
            config_.num_key_value_heads,
            config_.num_key_value_heads,
            config_.num_hidden_layers,
            config_.dtype,
            *paged_kv_cache_config,
            rank_info_);
    } else {
        throw std::runtime_error("Unsupported cache type");
    }
}

} // namespace infinilm::models::llama
