#include "last_logits_scope.hpp"
#include "../profiler.hpp"
#include "llama_for_causal_lm.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"
#include <iostream>

namespace infinilm::models::llama {

LlamaForCausalLM::LlamaForCausalLM(const LlamaConfig &config,
                                   const infinicore::Device &device,
                                   engine::distributed::RankInfo rank_info) {

    // Initialize module's device_ member
    device_ = device;

    const auto &dtype{config.dtype};

    // Initialize base model
    INFINICORE_NN_MODULE_INIT(model, config, device, rank_info);

    // Initialize language modeling head
    // Note: If tie_word_embeddings is true, we would share weights with embed_tokens
    // For now, we create a separate linear layer
    INFINICORE_NN_MODULE_INIT(lm_head, config.hidden_size, config.vocab_size, false,
                              dtype, device);
}

LlamaForCausalLM::Output LlamaForCausalLM::forward(const Input &input) const {
    auto input_ids = input.input_ids.value();
    auto position_ids = input.position_ids.value();
    auto past_sequence_lengths = input.past_sequence_lengths;
    auto total_sequence_length = input.total_sequence_lengths;
    auto input_offsets = input.input_offsets;
    auto block_tables = input.block_tables;
    auto slot_mapping = input.slot_mapping;

    // 1. Forward through base model to get hidden states
    auto hidden_states = model_->forward(
        input_ids, position_ids, past_sequence_lengths, total_sequence_length, input_offsets, block_tables, slot_mapping);

    // 2. Apply language modeling head to get logits
    auto logits = [&]() {
        const auto phase = infinilm::profile::phase_from_tensor(hidden_states);
        // This model wrapper does not store a rank; -1 means unspecified, not rank zero.
        INFINILM_PROFILE_SCOPE(phase, -1, -1, "model_lm_head", hidden_states);
        return iv_last_logits::project(lm_head_, hidden_states);
    }();

    return {logits};
}

void LlamaForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    model_->reset_cache(cache_config);
}

} // namespace infinilm::models::llama
