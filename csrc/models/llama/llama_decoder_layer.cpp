void iv_chain_layer(bool);
#include "llama_decoder_layer.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include "../profiler.hpp"

#include <optional>
std::pair<infinicore::Tensor, infinicore::Tensor> iv_add_rms_fusion(
    infinicore::Tensor, infinicore::Tensor, infinicore::Tensor, float);

namespace infinilm::models::llama {

LlamaDecoderLayer::LlamaDecoderLayer(const LlamaConfig &config,
                                     const infinicore::Device &device,
                                     size_t layer_idx,
                                     engine::distributed::RankInfo rank_info) : layer_idx_(layer_idx), rank_info_(rank_info) {
    const auto &dtype{config.dtype};

    // Initialize layer normalization layers
    INFINICORE_NN_MODULE_INIT(input_layernorm, config.hidden_size, config.rms_norm_eps,
                              dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, config.hidden_size, config.rms_norm_eps,
                              dtype, device);

    // Initialize attention and MLP modules
    INFINICORE_NN_MODULE_INIT(self_attn, config, device, layer_idx, rank_info_);
    INFINICORE_NN_MODULE_INIT(mlp, config, device, layer_idx, rank_info_);
}

infinicore::Tensor LlamaDecoderLayer::forward(const infinicore::Tensor &hidden_states,
                                              const infinicore::Tensor &position_ids,
                                              std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                              std::optional<infinicore::Tensor> past_sequence_lengths,
                                              std::optional<infinicore::Tensor> total_sequence_lengths,
                                              std::optional<infinicore::Tensor> input_offsets,
                                              std::optional<infinicore::Tensor> block_tables,
                                              std::optional<infinicore::Tensor> slot_mapping) const {
    const auto phase = infinilm::profile::phase_from_tensor(hidden_states);
    INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "decoder_layer_total", hidden_states);

    // Save residual for attention
    auto residual = hidden_states;

    // 1. Pre-attention layer normalization
    auto normed_states = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "input_layernorm", hidden_states);
        return input_layernorm_->forward(hidden_states);
    }();

    // 2. Self-attention with residual connection
    auto attn_output = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "self_attn_total_from_layer", normed_states);
        return self_attn_->forward(normed_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths, input_offsets, block_tables, slot_mapping);
    }();

    // One fused kernel; wrapper retains the separate operations when disabled.
    auto fused = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank,
                              "attn_add_rmsnorm", attn_output);
        return iv_add_rms_fusion(residual, attn_output,
            post_attention_layernorm_->weight(), static_cast<float>(post_attention_layernorm_->eps()));
    }();
    normed_states = fused.first;
    residual = fused.second;
    auto output = residual;

    // 4. MLP with residual connection
    auto mlp_output = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "mlp_total_from_layer", normed_states);
        return mlp_->forward(normed_states);
    }();

    // Add residual: output = output + mlp_output
    output = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "mlp_residual_add", mlp_output);
        return infinicore::op::add(residual, mlp_output);
    }();

    return output;
}

std::pair<infinicore::Tensor, infinicore::Tensor> LlamaDecoderLayer::forward_deferred(const infinicore::Tensor &hidden_states,
                                              const infinicore::Tensor &position_ids,
                                              std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                              std::optional<infinicore::Tensor> past_sequence_lengths,
                                              std::optional<infinicore::Tensor> total_sequence_lengths,
                                              std::optional<infinicore::Tensor> input_offsets,
                                              std::optional<infinicore::Tensor> block_tables,
                                              std::optional<infinicore::Tensor> slot_mapping,
                                              std::optional<infinicore::Tensor> pending_residual) const {
    const auto phase = infinilm::profile::phase_from_tensor(hidden_states);
    INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "decoder_layer_total", hidden_states);

    // Save residual for attention
    iv_chain_layer(pending_residual.has_value());
    auto residual = hidden_states;

    // 1. Pre-attention layer normalization
    auto normed_states = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "input_layernorm", hidden_states);
        if (pending_residual.has_value()) {
            auto norm = iv_add_rms_fusion(pending_residual.value(), hidden_states,
                input_layernorm_->weight(), static_cast<float>(input_layernorm_->eps()));
            residual = norm.second;
            return norm.first;
        }
        return input_layernorm_->forward(hidden_states);
    }();

    // 2. Self-attention with residual connection
    auto attn_output = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "self_attn_total_from_layer", normed_states);
        return self_attn_->forward(normed_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths, input_offsets, block_tables, slot_mapping);
    }();

    // One fused kernel; wrapper retains the separate operations when disabled.
    auto fused = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank,
                              "attn_add_rmsnorm", attn_output);
        return iv_add_rms_fusion(residual, attn_output,
            post_attention_layernorm_->weight(), static_cast<float>(post_attention_layernorm_->eps()));
    }();
    normed_states = fused.first;
    residual = fused.second;

    // 4. MLP with residual connection
    auto mlp_output = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "mlp_total_from_layer", normed_states);
        return mlp_->forward(normed_states);
    }();

    // Carry both values explicitly to the next norm; do not materialize this Add here.
    return {mlp_output, residual};
}

} // namespace infinilm::models::llama
