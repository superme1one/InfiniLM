#include "rope_pair_model_switch.hpp"
#include "decode_layout.hpp"
#include "llama_attention.hpp"
#include "infinicore/ops/gemm.hpp"

#include "../../utils.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mul.hpp"
#include "infiniop/ops/causal_softmax.h"
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include "../profiler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <fstream>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cstddef>
#include <string>
#include <limits>


extern "C" int infiniopAttentionStateFusedDecodeBf16Prototype(
    void *out,
    float *lse_out,
    const void *q,
    const void *k,
    const void *v,
    size_t kv_len,
    size_t batch_heads,
    size_t head_dim,
    ptrdiff_t q_stride,
    ptrdiff_t k_stride,
    ptrdiff_t v_stride,
    ptrdiff_t out_stride,
    float scale);

namespace {

#ifndef INFINILM_SHARED_PREFIX_KV_REUSE_ISOLATION_MODES
#define INFINILM_SHARED_PREFIX_KV_REUSE_ISOLATION_MODES 1

const char *infiniLmSharedPrefixIsolationMode() {
    const char *env = std::getenv("INFINILM_SHARED_PREFIX_KV_REUSE");
    return env == nullptr ? "" : env;
}

bool infiniLmSharedPrefixIsolationEnabled() {
    const char *mode = infiniLmSharedPrefixIsolationMode();
    return std::strcmp(mode, "qk_only") == 0 ||
           std::strcmp(mode, "v_only") == 0 ||
           std::strcmp(mode, "qk_v") == 0;
}

bool infiniLmSharedPrefixIsolationReuseQk() {
    const char *mode = infiniLmSharedPrefixIsolationMode();
    return std::strcmp(mode, "qk_only") == 0 ||
           std::strcmp(mode, "qk_v") == 0;
}

bool infiniLmSharedPrefixIsolationReuseV() {
    const char *mode = infiniLmSharedPrefixIsolationMode();
    return std::strcmp(mode, "v_only") == 0 ||
           std::strcmp(mode, "qk_v") == 0;
}

#endif // INFINILM_SHARED_PREFIX_KV_REUSE_ISOLATION_MODES


#ifndef INFINILM_SHARED_PREFIX_KV_REUSE_PATCH
#define INFINILM_SHARED_PREFIX_KV_REUSE_PATCH 1

bool infiniLmSharedPrefixKvReuseEnabled() {
    const char *env = std::getenv("INFINILM_SHARED_PREFIX_KV_REUSE");
    return env != nullptr && std::strcmp(env, "identical_batch") == 0;
}

bool infiniLmSharedPrefixIdenticalBatchContract() {
    const char *env = std::getenv("INFINILM_SHARED_PREFIX_IDENTICAL_BATCH");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool infiniLmSharedPrefixKvReuseTraceEnabled() {
    const char *env = std::getenv("INFINILM_SHARED_PREFIX_KV_REUSE_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

size_t infiniLmSharedPrefixKvReuseMinBatch() {
    const char *env = std::getenv("INFINILM_SHARED_PREFIX_KV_REUSE_MIN_BATCH");
    if (env == nullptr || env[0] == '\0') {
        return 4;
    }
    const size_t value = static_cast<size_t>(std::strtoull(env, nullptr, 10));
    return value > 1 ? value : 2;
}

size_t infiniLmSharedPrefixKvReuseMinKv() {
    const char *env = std::getenv("INFINILM_SHARED_PREFIX_KV_REUSE_MIN_KV");
    if (env == nullptr || env[0] == '\0') {
        return 1024;
    }
    return static_cast<size_t>(std::strtoull(env, nullptr, 10));
}

#endif // INFINILM_SHARED_PREFIX_KV_REUSE_PATCH

bool infiniLmKvPadTraceEnabled() {
    const char *env = std::getenv("INFINILM_KV_PAD_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}



size_t infiniLmKvBlockMassBlockSize();

float infiniLmKvBlockMassTraceValue(
    const std::byte *raw,
    infinicore::DataType dtype,
    size_t index);

bool infiniLmKvBlockHeadTraceEnabled() {
    const char *env = std::getenv("INFINILM_KV_BLOCK_HEAD_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool infiniLmKvBlockHeadTraceLayerSelected(size_t layer_idx) {
    const char *env = std::getenv("INFINILM_KV_BLOCK_HEAD_TRACE_LAYERS");
    if (env == nullptr || env[0] == '\0') {
        return layer_idx >= 4 && layer_idx <= 7;
    }
    const char *cursor = env;
    while (*cursor != '\0') {
        char *end = nullptr;
        const unsigned long long value = std::strtoull(cursor, &end, 10);
        if (end != cursor && static_cast<size_t>(value) == layer_idx) {
            return true;
        }
        if (end == cursor) {
            ++cursor;
        } else {
            cursor = end;
        }
        while (*cursor == ',' || *cursor == ' ' || *cursor == ';') {
            ++cursor;
        }
    }
    return false;
}

size_t infiniLmKvBlockHeadTraceTopK() {
    const char *env = std::getenv("INFINILM_KV_BLOCK_HEAD_TRACE_TOP_K");
    if (env == nullptr || env[0] == '\0') {
        return 16;
    }
    const size_t value = static_cast<size_t>(std::strtoull(env, nullptr, 10));
    return value == 0 ? 16 : value;
}

void infiniLmKvBlockHeadTrace(
    infinicore::Tensor attn_weight_softmax,
    size_t layer_idx,
    size_t real_total_seq_len,
    size_t total_seq_len,
    size_t batch_size,
    size_t num_heads) {
    if (!infiniLmKvBlockHeadTraceEnabled() ||
        !infiniLmKvBlockHeadTraceLayerSelected(layer_idx) ||
        real_total_seq_len == 0 ||
        total_seq_len == 0 ||
        batch_size == 0 ||
        num_heads == 0) {
        return;
    }

    const size_t block_size = infiniLmKvBlockMassBlockSize();
    const size_t blocks = (real_total_seq_len + block_size - 1) / block_size;
    const size_t top_k = std::min(infiniLmKvBlockHeadTraceTopK(), blocks);
    if (blocks == 0 || top_k == 0) {
        return;
    }

    static thread_local std::unordered_map<size_t, size_t> trace_steps;
    const size_t step = trace_steps[layer_idx]++;
    const char *tag_env = std::getenv("INFINILM_KV_BLOCK_MASS_TRACE_TAG");
    const char *tag = (tag_env == nullptr || tag_env[0] == '\0') ? "unknown" : tag_env;

    auto trace_cpu = attn_weight_softmax->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::syncStream();
    const std::byte *raw = trace_cpu->data();
    auto dtype = trace_cpu->dtype();

    // Prompts are duplicated inside a batch in the current probe, so tracing
    // batch 0 captures every attention head without repeating identical rows.
    const size_t batch_idx = 0;
    for (size_t head = 0; head < num_heads; ++head) {
        const size_t row = batch_idx * num_heads + head;
        const size_t base = row * total_seq_len;
        std::vector<double> block_mass(blocks, 0.0);
        double real_mass = 0.0;
        for (size_t col = 0; col < real_total_seq_len; ++col) {
            const double value = static_cast<double>(
                infiniLmKvBlockMassTraceValue(raw, dtype, base + col));
            block_mass[col / block_size] += value;
            real_mass += value;
        }

        std::vector<std::pair<double, size_t>> ranked;
        ranked.reserve(blocks);
        for (size_t block = 0; block < blocks; ++block) {
            ranked.emplace_back(block_mass[block], block);
        }
        std::sort(
            ranked.begin(),
            ranked.end(),
            [](const auto &left, const auto &right) {
                if (left.first != right.first) {
                    return left.first > right.first;
                }
                return left.second < right.second;
            });

        double top_mass = 0.0;
        double cumulative = 0.0;
        double entropy = 0.0;
        size_t blocks90 = blocks;
        size_t blocks95 = blocks;
        bool saw90 = false;
        bool saw95 = false;
        const double target90 = std::max(real_mass, 0.0) * 0.90;
        const double target95 = std::max(real_mass, 0.0) * 0.95;
        for (size_t i = 0; i < ranked.size(); ++i) {
            const double mass = std::max(ranked[i].first, 0.0);
            if (i < top_k) {
                top_mass += mass;
            }
            if (mass > 0.0) {
                entropy -= mass * std::log(mass);
            }
            cumulative += mass;
            if (!saw90 && cumulative >= target90) {
                blocks90 = i + 1;
                saw90 = true;
            }
            if (!saw95 && cumulative >= target95) {
                blocks95 = i + 1;
                saw95 = true;
            }
        }

        std::cerr << "INFINILM_KV_BLOCK_HEAD_TRACE"
                  << " tag=" << tag
                  << " step=" << step
                  << " layer=" << layer_idx
                  << " batch=" << batch_idx
                  << " head=" << head
                  << " real_len=" << real_total_seq_len
                  << " block_size=" << block_size
                  << " blocks=" << blocks
                  << " top_k=" << top_k
                  << " top_mass=" << top_mass
                  << " blocks90=" << blocks90
                  << " blocks95=" << blocks95
                  << " entropy=" << entropy
                  << " top_ids=";
        for (size_t i = 0; i < top_k; ++i) {
            if (i != 0) {
                std::cerr << ',';
            }
            std::cerr << ranked[i].second;
        }
        std::cerr << std::endl;
    }
}


size_t infiniLmKvBlockMassBlockSize();

float infiniLmKvBlockMassTraceValue(
    const std::byte *raw,
    infinicore::DataType dtype,
    size_t index);

bool infiniLmKvBlockMeanKeyTraceEnabled() {
    const char *env = std::getenv("INFINILM_KV_BLOCK_MEAN_KEY_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool infiniLmKvBlockMeanKeyTraceLayerSelected(size_t layer_idx) {
    const char *env = std::getenv("INFINILM_KV_BLOCK_MEAN_KEY_TRACE_LAYERS");
    if (env == nullptr || env[0] == '\0') {
        return layer_idx >= 4 && layer_idx <= 7;
    }
    const char *cursor = env;
    while (*cursor != '\0') {
        char *end = nullptr;
        const unsigned long long value = std::strtoull(cursor, &end, 10);
        if (end != cursor && static_cast<size_t>(value) == layer_idx) {
            return true;
        }
        cursor = end == cursor ? cursor + 1 : end;
        while (*cursor == ',' || *cursor == ' ' || *cursor == ';') {
            ++cursor;
        }
    }
    return false;
}

void infiniLmKvBlockMeanKeyTrace(
    infinicore::Tensor q,
    infinicore::Tensor k,
    infinicore::Tensor attn_weight_softmax,
    size_t layer_idx,
    size_t real_total_seq_len,
    size_t total_seq_len,
    size_t batch_size,
    size_t num_heads,
    size_t head_dim) {
    if (!infiniLmKvBlockMeanKeyTraceEnabled() ||
        !infiniLmKvBlockMeanKeyTraceLayerSelected(layer_idx) ||
        real_total_seq_len == 0 || total_seq_len == 0 ||
        batch_size == 0 || num_heads == 0 || head_dim == 0) {
        return;
    }

    const size_t block_size = infiniLmKvBlockMassBlockSize();
    const size_t blocks = (real_total_seq_len + block_size - 1) / block_size;
    if (blocks == 0) {
        return;
    }

    static thread_local std::unordered_map<size_t, size_t> trace_steps;
    const size_t step = trace_steps[layer_idx]++;
    const char *tag_env = std::getenv("INFINILM_KV_BLOCK_MASS_TRACE_TAG");
    const char *tag = (tag_env == nullptr || tag_env[0] == '\0') ? "unknown" : tag_env;

    auto q_cpu = q->contiguous()->to(infinicore::Device::cpu());
    auto k_cpu = k->contiguous()->to(infinicore::Device::cpu());
    auto p_cpu = attn_weight_softmax->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::syncStream();
    const std::byte *q_raw = q_cpu->data();
    const std::byte *k_raw = k_cpu->data();
    const std::byte *p_raw = p_cpu->data();
    const auto q_dtype = q_cpu->dtype();
    const auto k_dtype = k_cpu->dtype();
    const auto p_dtype = p_cpu->dtype();

    const size_t budgets[] = {16, 32, 48};
    const size_t batch_idx = 0;
    for (size_t head = 0; head < num_heads; ++head) {
        const size_t row = batch_idx * num_heads + head;
        std::vector<double> true_mass(blocks, 0.0);
        std::vector<std::pair<double, size_t>> scorer_ranked;
        scorer_ranked.reserve(blocks);
        double real_mass = 0.0;

        for (size_t block = 0; block < blocks; ++block) {
            const size_t token_begin = block * block_size;
            const size_t token_end = std::min(token_begin + block_size, real_total_seq_len);
            const size_t token_count = token_end - token_begin;
            double score = 0.0;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                double mean_k = 0.0;
                for (size_t token = token_begin; token < token_end; ++token) {
                    const size_t k_index = (row * total_seq_len + token) * head_dim + dim;
                    mean_k += static_cast<double>(
                        infiniLmKvBlockMassTraceValue(k_raw, k_dtype, k_index));
                }
                mean_k /= static_cast<double>(token_count);
                const size_t q_index = row * head_dim + dim;
                const double q_value = static_cast<double>(
                    infiniLmKvBlockMassTraceValue(q_raw, q_dtype, q_index));
                score += q_value * mean_k;
            }
            scorer_ranked.emplace_back(score, block);

            double mass = 0.0;
            const size_t p_base = row * total_seq_len;
            for (size_t token = token_begin; token < token_end; ++token) {
                mass += static_cast<double>(
                    infiniLmKvBlockMassTraceValue(p_raw, p_dtype, p_base + token));
            }
            true_mass[block] = mass;
            real_mass += mass;
        }

        std::sort(
            scorer_ranked.begin(), scorer_ranked.end(),
            [](const auto &left, const auto &right) {
                if (left.first != right.first) {
                    return left.first > right.first;
                }
                return left.second < right.second;
            });
        std::vector<std::pair<double, size_t>> oracle_ranked;
        oracle_ranked.reserve(blocks);
        for (size_t block = 0; block < blocks; ++block) {
            oracle_ranked.emplace_back(true_mass[block], block);
        }
        std::sort(
            oracle_ranked.begin(), oracle_ranked.end(),
            [](const auto &left, const auto &right) {
                if (left.first != right.first) {
                    return left.first > right.first;
                }
                return left.second < right.second;
            });

        for (size_t requested_budget : budgets) {
            const size_t budget = std::min(requested_budget, blocks);
            for (size_t variant = 0; variant < 2; ++variant) {
                std::vector<bool> selected(blocks, false);
                size_t selected_count = 0;
                if (variant == 1) {
                    selected[0] = true;
                    selected_count = 1;
                    const size_t recent = std::min<size_t>(4, blocks);
                    for (size_t i = blocks - recent; i < blocks; ++i) {
                        if (!selected[i]) {
                            selected[i] = true;
                            ++selected_count;
                        }
                    }
                }
                for (const auto &entry : scorer_ranked) {
                    if (selected_count >= budget) {
                        break;
                    }
                    if (!selected[entry.second]) {
                        selected[entry.second] = true;
                        ++selected_count;
                    }
                }

                double retained_mass = 0.0;
                size_t oracle_overlap = 0;
                for (size_t block = 0; block < blocks; ++block) {
                    if (selected[block]) {
                        retained_mass += true_mass[block];
                    }
                }
                for (size_t i = 0; i < budget; ++i) {
                    if (selected[oracle_ranked[i].second]) {
                        ++oracle_overlap;
                    }
                }
                const double normalized_mass = real_mass > 0.0 ? retained_mass / real_mass : 0.0;
                const double oracle_recall = budget > 0
                    ? static_cast<double>(oracle_overlap) / static_cast<double>(budget)
                    : 0.0;
                std::cerr << "INFINILM_KV_BLOCK_MEAN_KEY_SCORER_TRACE"
                          << " tag=" << tag
                          << " step=" << step
                          << " layer=" << layer_idx
                          << " head=" << head
                          << " blocks=" << blocks
                          << " budget=" << budget
                          << " variant=" << (variant == 0 ? "mean" : "mean_sink_recent")
                          << " retained_mass=" << normalized_mass
                          << " oracle_recall=" << oracle_recall
                          << std::endl;
            }
        }
    }
}


size_t infiniLmKvBlockMassBlockSize();

float infiniLmKvBlockMassTraceValue(
    const std::byte *raw,
    infinicore::DataType dtype,
    size_t index);

bool infiniLmKvBlockMultiRepTraceEnabled() {
    const char *env = std::getenv("INFINILM_KV_BLOCK_MULTI_REP_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool infiniLmKvBlockMultiRepTraceLayerSelected(size_t layer_idx) {
    const char *env = std::getenv("INFINILM_KV_BLOCK_MULTI_REP_TRACE_LAYERS");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    const char *cursor = env;
    while (*cursor != '\0') {
        char *end = nullptr;
        const unsigned long long value = std::strtoull(cursor, &end, 10);
        if (end != cursor && static_cast<size_t>(value) == layer_idx) {
            return true;
        }
        cursor = end == cursor ? cursor + 1 : end;
        while (*cursor == ',' || *cursor == ' ' || *cursor == ';') {
            ++cursor;
        }
    }
    return false;
}

double infiniLmKvBlockMultiRepScore(
    const std::byte *q_raw,
    infinicore::DataType q_dtype,
    const std::byte *k_raw,
    infinicore::DataType k_dtype,
    size_t row,
    size_t total_seq_len,
    size_t head_dim,
    size_t token_begin,
    size_t token_end,
    size_t representatives,
    double scaling) {
    std::vector<std::pair<double, size_t>> segment_scores;
    const size_t token_count = token_end - token_begin;
    for (size_t segment = 0; segment < representatives; ++segment) {
        const size_t begin = token_begin + token_count * segment / representatives;
        const size_t end = token_begin + token_count * (segment + 1) / representatives;
        if (end <= begin) {
            continue;
        }
        double dot = 0.0;
        for (size_t dim = 0; dim < head_dim; ++dim) {
            double mean_k = 0.0;
            for (size_t token = begin; token < end; ++token) {
                const size_t index = (row * total_seq_len + token) * head_dim + dim;
                mean_k += static_cast<double>(
                    infiniLmKvBlockMassTraceValue(k_raw, k_dtype, index));
            }
            mean_k /= static_cast<double>(end - begin);
            const double q_value = static_cast<double>(
                infiniLmKvBlockMassTraceValue(
                    q_raw, q_dtype, row * head_dim + dim));
            dot += q_value * mean_k;
        }
        segment_scores.emplace_back(dot * scaling, end - begin);
    }
    if (segment_scores.empty()) {
        return -std::numeric_limits<double>::infinity();
    }
    double max_score = segment_scores[0].first;
    for (const auto &entry : segment_scores) {
        max_score = std::max(max_score, entry.first);
    }
    double exp_sum = 0.0;
    for (const auto &entry : segment_scores) {
        exp_sum += static_cast<double>(entry.second) * std::exp(entry.first - max_score);
    }
    return max_score + std::log(exp_sum);
}

void infiniLmKvBlockMultiRepTrace(
    infinicore::Tensor q,
    infinicore::Tensor k,
    infinicore::Tensor attn_weight_softmax,
    size_t layer_idx,
    size_t real_total_seq_len,
    size_t total_seq_len,
    size_t batch_size,
    size_t num_heads,
    size_t head_dim,
    double scaling) {
    if (!infiniLmKvBlockMultiRepTraceEnabled() ||
        !infiniLmKvBlockMultiRepTraceLayerSelected(layer_idx) ||
        real_total_seq_len == 0 || total_seq_len == 0 ||
        batch_size == 0 || num_heads == 0 || head_dim == 0) {
        return;
    }
    const size_t block_size = infiniLmKvBlockMassBlockSize();
    const size_t blocks = (real_total_seq_len + block_size - 1) / block_size;
    if (blocks == 0) {
        return;
    }

    static thread_local std::unordered_map<size_t, size_t> trace_steps;
    const size_t step = trace_steps[layer_idx]++;
    const char *tag_env = std::getenv("INFINILM_KV_BLOCK_MASS_TRACE_TAG");
    const char *tag = (tag_env == nullptr || tag_env[0] == '\0') ? "unknown" : tag_env;

    auto q_cpu = q->contiguous()->to(infinicore::Device::cpu());
    auto k_cpu = k->contiguous()->to(infinicore::Device::cpu());
    auto p_cpu = attn_weight_softmax->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::syncStream();
    const std::byte *q_raw = q_cpu->data();
    const std::byte *k_raw = k_cpu->data();
    const std::byte *p_raw = p_cpu->data();
    const auto q_dtype = q_cpu->dtype();
    const auto k_dtype = k_cpu->dtype();
    const auto p_dtype = p_cpu->dtype();
    const size_t budgets[] = {16, 32, 48};
    const size_t representative_counts[] = {2, 4};
    const size_t batch_idx = 0;

    for (size_t head = 0; head < num_heads; ++head) {
        const size_t row = batch_idx * num_heads + head;
        std::vector<double> true_mass(blocks, 0.0);
        double real_mass = 0.0;
        for (size_t block = 0; block < blocks; ++block) {
            const size_t begin = block * block_size;
            const size_t end = std::min(begin + block_size, real_total_seq_len);
            double mass = 0.0;
            for (size_t token = begin; token < end; ++token) {
                mass += static_cast<double>(infiniLmKvBlockMassTraceValue(
                    p_raw, p_dtype, row * total_seq_len + token));
            }
            true_mass[block] = mass;
            real_mass += mass;
        }

        std::vector<std::pair<double, size_t>> oracle_ranked;
        for (size_t block = 0; block < blocks; ++block) {
            oracle_ranked.emplace_back(true_mass[block], block);
        }
        std::sort(oracle_ranked.begin(), oracle_ranked.end(), [](const auto &a, const auto &b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });

        for (size_t representatives : representative_counts) {
            std::vector<std::pair<double, size_t>> ranked;
            ranked.reserve(blocks);
            for (size_t block = 0; block < blocks; ++block) {
                const size_t begin = block * block_size;
                const size_t end = std::min(begin + block_size, real_total_seq_len);
                ranked.emplace_back(
                    infiniLmKvBlockMultiRepScore(
                        q_raw, q_dtype, k_raw, k_dtype, row, total_seq_len,
                        head_dim, begin, end, representatives, scaling),
                    block);
            }
            std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });

            for (size_t requested_budget : budgets) {
                const size_t budget = std::min(requested_budget, blocks);
                std::vector<bool> selected(blocks, false);
                selected[0] = true;
                size_t selected_count = 1;
                const size_t recent = std::min<size_t>(4, blocks);
                for (size_t block = blocks - recent; block < blocks; ++block) {
                    if (!selected[block]) {
                        selected[block] = true;
                        ++selected_count;
                    }
                }
                for (const auto &entry : ranked) {
                    if (selected_count >= budget) {
                        break;
                    }
                    if (!selected[entry.second]) {
                        selected[entry.second] = true;
                        ++selected_count;
                    }
                }

                double retained_mass = 0.0;
                size_t oracle_overlap = 0;
                for (size_t block = 0; block < blocks; ++block) {
                    if (selected[block]) {
                        retained_mass += true_mass[block];
                    }
                }
                for (size_t i = 0; i < budget; ++i) {
                    if (selected[oracle_ranked[i].second]) {
                        ++oracle_overlap;
                    }
                }
                std::cerr << "INFINILM_KV_BLOCK_MEAN_KEY_SCORER_TRACE"
                          << " tag=" << tag
                          << " step=" << step
                          << " layer=" << layer_idx
                          << " head=" << head
                          << " blocks=" << blocks
                          << " budget=" << budget
                          << " variant=submean" << representatives << "_sink_recent"
                          << " retained_mass=" << (real_mass > 0.0 ? retained_mass / real_mass : 0.0)
                          << " oracle_recall="
                          << (budget > 0 ? static_cast<double>(oracle_overlap) / budget : 0.0)
                          << std::endl;
            }
        }
    }
}



// INFINILM_KV_CACHE_MIXED_QUANT_TRACE
struct InfiniLmKvCacheQuantConfig {
    bool enabled = false;
    bool quantize_k = false;
    bool quantize_v = false;
    int bits = 8;
    size_t group_size = 32;
    size_t residual_tokens = 128;
    std::string mode = "off";
};

InfiniLmKvCacheQuantConfig infiniLmKvCacheQuantConfig() {
    InfiniLmKvCacheQuantConfig config;
    const char *mode_env = std::getenv("INFINILM_KV_CACHE_QUANT_MODE");
    if (mode_env == nullptr || mode_env[0] == '\0' || std::string(mode_env) == "off") {
        return config;
    }

    config.mode = std::string(mode_env);
    config.quantize_k = config.mode.rfind("k_pc_int8_r", 0) == 0 ||
                        config.mode.rfind("kv_mixed_int8_r", 0) == 0;
    config.quantize_v = config.mode.rfind("v_pt_int8_r", 0) == 0 ||
                        config.mode.rfind("kv_mixed_int8_r", 0) == 0;
    if (!config.quantize_k && !config.quantize_v) {
        return InfiniLmKvCacheQuantConfig{};
    }

    const char *group_env = std::getenv("INFINILM_KV_CACHE_QUANT_GROUP_SIZE");
    if (group_env != nullptr && group_env[0] != '\0') {
        config.group_size = static_cast<size_t>(std::strtoull(group_env, nullptr, 10));
    }
    if (config.group_size == 0) {
        config.group_size = 32;
    }
    const char *residual_env = std::getenv("INFINILM_KV_CACHE_QUANT_RESIDUAL_TOKENS");
    if (residual_env != nullptr && residual_env[0] != '\0') {
        config.residual_tokens = static_cast<size_t>(std::strtoull(residual_env, nullptr, 10));
    }
    config.enabled = true;
    return config;
}

bool infiniLmKvCacheQuantTraceEnabled() {
    const char *env = std::getenv("INFINILM_KV_CACHE_QUANT_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

size_t infiniLmKvCacheQuantTraceLayerLimit() {
    const char *env = std::getenv("INFINILM_KV_CACHE_QUANT_TRACE_LAYER_LIMIT");
    if (env == nullptr || env[0] == '\0') {
        return 4;
    }
    return static_cast<size_t>(std::strtoull(env, nullptr, 10));
}

std::string infiniLmKvCacheQuantEnv(const char *name, const char *fallback) {
    const char *env = std::getenv(name);
    return (env == nullptr || env[0] == '\0') ? std::string(fallback) : std::string(env);
}

size_t infiniLmKvCacheQuantTraceStep() {
    const char *env = std::getenv("INFINILM_TEACHER_FORCED_TRACE_STEP");
    if (env == nullptr || env[0] == '\0') {
        return 0;
    }
    return static_cast<size_t>(std::strtoull(env, nullptr, 10));
}

float infiniLmKvCacheBf16ToFloat(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t infiniLmKvCacheFloatToBf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

infinicore::Tensor infiniLmKvCacheMixedQuantizeDequantize(
    infinicore::Tensor input,
    const InfiniLmKvCacheQuantConfig &config,
    size_t layer_idx,
    const char *tensor_name,
    const char *phase,
    size_t batch_size,
    size_t kv_heads,
    size_t total_seq_len,
    size_t active_seq_len,
    size_t head_dim) {
    if (!config.enabled || input->dtype() != infinicore::DataType::BF16 || input->numel() == 0) {
        return input;
    }

    const size_t expected = batch_size * kv_heads * total_seq_len * head_dim;
    if (input->numel() != expected) {
        throw std::runtime_error("KV mixed quantization received an unexpected tensor layout");
    }

    auto cpu = input->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::syncStream();
    auto *raw = reinterpret_cast<uint16_t *>(cpu->data());
    const size_t count = cpu->numel();
    std::vector<uint16_t> reference(raw, raw + count);
    const size_t group_size = std::max<size_t>(1, config.group_size);
    const size_t active = std::min(active_seq_len, total_seq_len);
    const size_t quantized_tokens =
        active > config.residual_tokens ? active - config.residual_tokens : 0;
    const int qmax = 127;
    size_t groups = 0;
    size_t quantized_count = 0;

    auto offset = [=](size_t batch, size_t head, size_t token, size_t dim) {
        return (((batch * kv_heads + head) * total_seq_len + token) * head_dim + dim);
    };
    auto quantize_indices = [&](const std::vector<size_t> &indices) {
        if (indices.empty()) {
            return;
        }
        float max_abs = 0.0f;
        for (size_t index : indices) {
            max_abs = std::max(max_abs, std::abs(infiniLmKvCacheBf16ToFloat(raw[index])));
        }
        const float scale = max_abs > 0.0f ? max_abs / static_cast<float>(qmax) : 1.0f;
        ++groups;
        quantized_count += indices.size();
        for (size_t index : indices) {
            const float value = infiniLmKvCacheBf16ToFloat(raw[index]);
            int quantized = static_cast<int>(std::nearbyint(value / scale));
            quantized = std::max(-qmax, std::min(qmax, quantized));
            raw[index] = infiniLmKvCacheFloatToBf16(static_cast<float>(quantized) * scale);
        }
    };

    if (std::string(tensor_name) == "K") {
        // Per-channel K: each scale covers a group of tokens for one head_dim channel.
        for (size_t batch = 0; batch < batch_size; ++batch) {
            for (size_t head = 0; head < kv_heads; ++head) {
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    for (size_t begin = 0; begin < quantized_tokens; begin += group_size) {
                        const size_t end = std::min(quantized_tokens, begin + group_size);
                        std::vector<size_t> indices;
                        indices.reserve(end - begin);
                        for (size_t token = begin; token < end; ++token) {
                            indices.push_back(offset(batch, head, token, dim));
                        }
                        quantize_indices(indices);
                    }
                }
            }
        }
    } else {
        // Per-token V: each scale covers a head_dim group for one KV token.
        for (size_t batch = 0; batch < batch_size; ++batch) {
            for (size_t head = 0; head < kv_heads; ++head) {
                for (size_t token = 0; token < quantized_tokens; ++token) {
                    for (size_t begin = 0; begin < head_dim; begin += group_size) {
                        const size_t end = std::min(head_dim, begin + group_size);
                        std::vector<size_t> indices;
                        indices.reserve(end - begin);
                        for (size_t dim = begin; dim < end; ++dim) {
                            indices.push_back(offset(batch, head, token, dim));
                        }
                        quantize_indices(indices);
                    }
                }
            }
        }
    }

    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double max_error = 0.0;
    double dot = 0.0;
    double norm_ref = 0.0;
    double norm_candidate = 0.0;
    for (size_t index = 0; index < count; ++index) {
        const double lhs = static_cast<double>(infiniLmKvCacheBf16ToFloat(reference[index]));
        const double rhs = static_cast<double>(infiniLmKvCacheBf16ToFloat(raw[index]));
        const double error = std::abs(lhs - rhs);
        sum_abs += error;
        sum_sq += error * error;
        max_error = std::max(max_error, error);
        dot += lhs * rhs;
        norm_ref += lhs * lhs;
        norm_candidate += rhs * rhs;
    }

    if (infiniLmKvCacheQuantTraceEnabled() &&
        layer_idx < infiniLmKvCacheQuantTraceLayerLimit()) {
        const std::string tag = infiniLmKvCacheQuantEnv(
            "INFINILM_TEACHER_FORCED_TRACE_TAG", config.mode.c_str());
        const double denom = static_cast<double>(count);
        const double cosine = dot / std::sqrt(std::max(norm_ref * norm_candidate, 1.0e-30));
        const double storage_bits =
            static_cast<double>(quantized_count * 8) +
            static_cast<double>(groups * 16) +
            static_cast<double>((count - quantized_count) * 16);
        const double bytes_ratio = storage_bits / static_cast<double>(count * 16);
        std::cerr << "INFINILM_KV_CACHE_QUANT_TRACE"
                  << " tag=" << tag
                  << " step=" << infiniLmKvCacheQuantTraceStep()
                  << " phase=" << phase
                  << " layer=" << layer_idx
                  << " tensor=" << tensor_name
                  << " bits=" << config.bits
                  << " group_size=" << group_size
                  << " count=" << count
                  << " groups=" << groups
                  << " mean_abs=" << (sum_abs / denom)
                  << " max_abs=" << max_error
                  << " rmse=" << std::sqrt(sum_sq / denom)
                  << " cosine=" << cosine
                  << " bytes_ratio=" << bytes_ratio
                  << " granularity=" << (std::string(tensor_name) == "K" ? "per_channel" : "per_token")
                  << " residual_tokens=" << config.residual_tokens
                  << " quantized_count=" << quantized_count
                  << std::endl;
    }

    return cpu->to(input->device());
}

bool infiniLmKvBlockMassTraceEnabled() {
    const char *env = std::getenv("INFINILM_KV_BLOCK_MASS_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

size_t infiniLmKvBlockMassBlockSize() {
    const char *env = std::getenv("INFINILM_KV_BLOCK_MASS_BLOCK_SIZE");
    if (env == nullptr || env[0] == '\0') {
        return 16;
    }
    const size_t value = static_cast<size_t>(std::strtoull(env, nullptr, 10));
    return value == 0 ? 16 : value;
}

size_t infiniLmKvBlockMassLayerLimit() {
    const char *env = std::getenv("INFINILM_KV_BLOCK_MASS_LAYER_LIMIT");
    if (env == nullptr || env[0] == '\0') {
        return 8;
    }
    return static_cast<size_t>(std::strtoull(env, nullptr, 10));
}

float infiniLmKvBlockMassHalfToFloat(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    int32_t exp = static_cast<int32_t>((h >> 10) & 0x1fu);
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | (static_cast<uint32_t>(exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | (static_cast<uint32_t>(exp + 112) << 23) | (mant << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float infiniLmKvBlockMassTraceValue(
    const std::byte *raw,
    infinicore::DataType dtype,
    size_t index) {
    if (dtype == infinicore::DataType::F32) {
        return reinterpret_cast<const float *>(raw)[index];
    }
    uint16_t bits16 = reinterpret_cast<const uint16_t *>(raw)[index];
    if (dtype == infinicore::DataType::BF16) {
        uint32_t bits32 = static_cast<uint32_t>(bits16) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits32, sizeof(value));
        return value;
    }
    return infiniLmKvBlockMassHalfToFloat(bits16);
}

void infiniLmKvBlockMassTrace(
    infinicore::Tensor attn_weight_softmax,
    size_t layer_idx,
    size_t real_total_seq_len,
    size_t total_seq_len,
    size_t rows) {
    if (!infiniLmKvBlockMassTraceEnabled() ||
        layer_idx >= infiniLmKvBlockMassLayerLimit() ||
        real_total_seq_len == 0 ||
        total_seq_len == 0 ||
        rows == 0) {
        return;
    }

    const size_t block_size = infiniLmKvBlockMassBlockSize();
    const size_t blocks = (real_total_seq_len + block_size - 1) / block_size;
    if (blocks == 0) {
        return;
    }

    static thread_local std::unordered_map<size_t, size_t> trace_steps;
    const size_t step = trace_steps[layer_idx]++;
    const char *tag_env = std::getenv("INFINILM_KV_BLOCK_MASS_TRACE_TAG");
    const char *tag = (tag_env == nullptr || tag_env[0] == '\0') ? "unknown" : tag_env;

    auto trace_cpu = attn_weight_softmax->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::syncStream();
    const std::byte *raw = trace_cpu->data();
    auto dtype = trace_cpu->dtype();

    double sum_top1 = 0.0;
    double sum_top2 = 0.0;
    double sum_top4 = 0.0;
    double sum_top8 = 0.0;
    double sum_top16 = 0.0;
    double sum_top32 = 0.0;
    double min_top8 = 1.0;
    double min_top16 = 1.0;
    double sum_blocks90 = 0.0;
    double sum_blocks95 = 0.0;
    double sum_entropy = 0.0;
    double sum_real_mass = 0.0;

    auto top_mass = [](const std::vector<double> &values, size_t keep) {
        double total = 0.0;
        const size_t limit = std::min(keep, values.size());
        for (size_t i = 0; i < limit; ++i) {
            total += values[i];
        }
        return total;
    };

    for (size_t row = 0; row < rows; ++row) {
        std::vector<double> block_mass(blocks, 0.0);
        const size_t base = row * total_seq_len;
        for (size_t col = 0; col < real_total_seq_len; ++col) {
            const size_t block = col / block_size;
            const double value = static_cast<double>(
                infiniLmKvBlockMassTraceValue(raw, dtype, base + col));
            block_mass[block] += value;
            sum_real_mass += value;
        }
        std::sort(block_mass.begin(), block_mass.end(), std::greater<double>());

        const double top1 = top_mass(block_mass, 1);
        const double top2 = top_mass(block_mass, 2);
        const double top4 = top_mass(block_mass, 4);
        const double top8 = top_mass(block_mass, 8);
        const double top16 = top_mass(block_mass, 16);
        const double top32 = top_mass(block_mass, 32);
        sum_top1 += top1;
        sum_top2 += top2;
        sum_top4 += top4;
        sum_top8 += top8;
        sum_top16 += top16;
        sum_top32 += top32;
        min_top8 = std::min(min_top8, top8);
        min_top16 = std::min(min_top16, top16);

        double cumulative = 0.0;
        size_t blocks90 = blocks;
        size_t blocks95 = blocks;
        bool saw90 = false;
        bool saw95 = false;
        double entropy = 0.0;
        for (size_t i = 0; i < block_mass.size(); ++i) {
            const double mass = std::max(block_mass[i], 0.0);
            if (mass > 0.0) {
                entropy -= mass * std::log(mass);
            }
            cumulative += mass;
            if (!saw90 && cumulative >= 0.90) {
                blocks90 = i + 1;
                saw90 = true;
            }
            if (!saw95 && cumulative >= 0.95) {
                blocks95 = i + 1;
                saw95 = true;
            }
        }
        sum_blocks90 += static_cast<double>(blocks90);
        sum_blocks95 += static_cast<double>(blocks95);
        sum_entropy += entropy;
    }

    const double denom = static_cast<double>(rows);
    std::cerr << "INFINILM_KV_BLOCK_MASS_TRACE"
              << " tag=" << tag
              << " step=" << step
              << " layer=" << layer_idx
              << " rows=" << rows
              << " real_len=" << real_total_seq_len
              << " total_len=" << total_seq_len
              << " block_size=" << block_size
              << " blocks=" << blocks
              << " mean_top1=" << (sum_top1 / denom)
              << " mean_top2=" << (sum_top2 / denom)
              << " mean_top4=" << (sum_top4 / denom)
              << " mean_top8=" << (sum_top8 / denom)
              << " mean_top16=" << (sum_top16 / denom)
              << " mean_top32=" << (sum_top32 / denom)
              << " min_top8=" << min_top8
              << " min_top16=" << min_top16
              << " mean_blocks90=" << (sum_blocks90 / denom)
              << " mean_blocks95=" << (sum_blocks95 / denom)
              << " mean_entropy=" << (sum_entropy / denom)
              << " mean_real_mass=" << (sum_real_mass / denom)
              << std::endl;
}

bool infiniLmKvPadNumericTraceEnabled() {
    const char *env = std::getenv("INFINILM_KV_PAD_NUMERIC_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool infiniLmKvPadQkOnlyEnabled() {
    const char *env = std::getenv("INFINILM_KV_PAD_MODE");
    return env != nullptr &&
           (std::strcmp(env, "qk_only") == 0 ||
            std::strcmp(env, "qk_split") == 0 ||
            std::strcmp(env, "qk_fixed") == 0 ||
            std::strcmp(env, "qk_fixed_v_split") == 0);
}

bool infiniLmKvPadQkSplitEnabled() {
    const char *env = std::getenv("INFINILM_KV_PAD_MODE");
    return env != nullptr && std::strcmp(env, "qk_split") == 0;
}

bool infiniLmKvPadQkFixedEnabled() {
    const char *env = std::getenv("INFINILM_KV_PAD_MODE");
    return env != nullptr &&
           (std::strcmp(env, "qk_fixed") == 0 ||
            std::strcmp(env, "qk_fixed_v_split") == 0);
}


bool infiniLmKvPadVSplitBetaAccumEnabled() {
    const char *env = std::getenv("INFINILM_KV_PAD_V_SPLIT_ACCUM");
    return env != nullptr && std::strcmp(env, "beta") == 0;
}

bool infiniLmKvPadVSplitEnabled() {
    const char *env = std::getenv("INFINILM_KV_PAD_MODE");
    return env != nullptr && std::strcmp(env, "qk_fixed_v_split") == 0;
}




bool infiniLmAttentionLocalCertTraceEnabled() {
    const char *env = std::getenv("INFINILM_ATTENTION_LOCAL_CERT_TRACE");
    const char *teacher_env = std::getenv("INFINILM_TEACHER_FORCED_LOGITS_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0' &&
           teacher_env != nullptr && teacher_env[0] != '\0' && teacher_env[0] != '0';
}

size_t infiniLmAttentionLocalCertLayerLimit() {
    const char *env = std::getenv("INFINILM_ATTENTION_LOCAL_CERT_LAYER_LIMIT");
    if (env == nullptr || env[0] == '\0') {
        return 64;
    }
    return static_cast<size_t>(std::strtoull(env, nullptr, 10));
}

std::string infiniLmAttentionLocalCertEnv(const char *name, const char *fallback) {
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return std::string(fallback);
    }
    return std::string(env);
}

size_t infiniLmAttentionLocalCertTraceStep() {
    const char *env = std::getenv("INFINILM_TEACHER_FORCED_TRACE_STEP");
    if (env == nullptr || env[0] == '\0') {
        return 0;
    }
    return static_cast<size_t>(std::strtoull(env, nullptr, 10));
}

uint64_t infiniLmAttentionLocalCertKey(size_t layer_idx, size_t step) {
    return (static_cast<uint64_t>(layer_idx) << 32) | static_cast<uint64_t>(step);
}

float infiniLmAttentionLocalCertHalfToFloat(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    int32_t exp = static_cast<int32_t>((h >> 10) & 0x1fu);
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | (static_cast<uint32_t>(exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | (static_cast<uint32_t>(exp + 112) << 23) | (mant << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float infiniLmAttentionLocalCertValue(
    const std::byte *raw,
    infinicore::DataType dtype,
    size_t index) {
    if (dtype == infinicore::DataType::F32) {
        return reinterpret_cast<const float *>(raw)[index];
    }
    uint16_t bits16 = reinterpret_cast<const uint16_t *>(raw)[index];
    if (dtype == infinicore::DataType::BF16) {
        uint32_t bits32 = static_cast<uint32_t>(bits16) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits32, sizeof(value));
        return value;
    }
    return infiniLmAttentionLocalCertHalfToFloat(bits16);
}

void infiniLmAttentionLocalCertTrace(
    infinicore::Tensor tensor,
    size_t layer_idx,
    size_t real_total_seq_len,
    size_t total_seq_len,
    size_t batch_heads,
    size_t head_dim) {
    if (!infiniLmAttentionLocalCertTraceEnabled() ||
        layer_idx >= infiniLmAttentionLocalCertLayerLimit()) {
        return;
    }

    const std::string role = infiniLmAttentionLocalCertEnv(
        "INFINILM_TEACHER_FORCED_TRACE_ROLE",
        "candidate");
    const std::string tag = infiniLmAttentionLocalCertEnv(
        "INFINILM_TEACHER_FORCED_TRACE_TAG",
        "unknown");
    const size_t step = infiniLmAttentionLocalCertTraceStep();
    const uint64_t key = infiniLmAttentionLocalCertKey(layer_idx, step);

    auto cpu = tensor->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::syncStream();
    const std::byte *raw = cpu->data();
    auto dtype = cpu->dtype();
    const size_t count = cpu->numel();
    std::vector<float> values(count, 0.0f);
    for (size_t index = 0; index < count; ++index) {
        values[index] = infiniLmAttentionLocalCertValue(raw, dtype, index);
    }

    static thread_local std::unordered_map<uint64_t, std::vector<float>>
        attention_local_baselines;

    if (role == "baseline") {
        attention_local_baselines[key] = std::move(values);
        std::cerr << "INFINILM_ATTENTION_LOCAL_CERT_BASELINE"
                  << " tag=" << tag
                  << " step=" << step
                  << " layer=" << layer_idx
                  << " count=" << count
                  << " rows=" << batch_heads
                  << " head_dim=" << head_dim
                  << " real_len=" << real_total_seq_len
                  << " total_len=" << total_seq_len
                  << std::endl;
        return;
    }

    auto baseline_it = attention_local_baselines.find(key);
    if (baseline_it == attention_local_baselines.end() ||
        baseline_it->second.size() != values.size()) {
        std::cerr << "INFINILM_ATTENTION_LOCAL_CERT_MISSING"
                  << " tag=" << tag
                  << " step=" << step
                  << " layer=" << layer_idx
                  << " candidate_count=" << values.size()
                  << std::endl;
        return;
    }

    const auto &reference = baseline_it->second;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double max_abs = 0.0;
    double dot = 0.0;
    double norm_ref = 0.0;
    double norm_candidate = 0.0;
    size_t unequal = 0;

    double worst_row_rel = -1.0;
    double worst_row_delta_norm = 0.0;
    double worst_row_ref_norm = 0.0;
    double worst_row_rmse = 0.0;
    double worst_row_max_abs = 0.0;
    double row_cosine_min = 1.0;
    size_t worst_row = 0;
    const size_t safe_head_dim = std::max<size_t>(head_dim, 1);
    const size_t rows = count / safe_head_dim;

    for (size_t row = 0; row < rows; ++row) {
        double row_sum_sq = 0.0;
        double row_ref_norm_sq = 0.0;
        double row_candidate_norm_sq = 0.0;
        double row_dot = 0.0;
        double row_max_abs = 0.0;
        const size_t base = row * safe_head_dim;
        for (size_t col = 0; col < safe_head_dim && base + col < count; ++col) {
            const size_t index = base + col;
            const double lhs = static_cast<double>(reference[index]);
            const double rhs = static_cast<double>(values[index]);
            const double diff = lhs - rhs;
            const double abs_diff = std::abs(diff);
            unequal += abs_diff != 0.0 ? 1 : 0;
            sum_abs += abs_diff;
            sum_sq += diff * diff;
            max_abs = std::max(max_abs, abs_diff);
            dot += lhs * rhs;
            norm_ref += lhs * lhs;
            norm_candidate += rhs * rhs;

            row_sum_sq += diff * diff;
            row_ref_norm_sq += lhs * lhs;
            row_candidate_norm_sq += rhs * rhs;
            row_dot += lhs * rhs;
            row_max_abs = std::max(row_max_abs, abs_diff);
        }
        const double row_delta_norm = std::sqrt(row_sum_sq);
        const double row_ref_norm = std::sqrt(row_ref_norm_sq);
        const double row_rel = row_delta_norm / std::max(row_ref_norm, 1.0e-12);
        const double row_rmse = std::sqrt(row_sum_sq / safe_head_dim);
        double row_cosine = 0.0;
        if (row_ref_norm_sq > 0.0 && row_candidate_norm_sq > 0.0) {
            row_cosine = row_dot / std::sqrt(row_ref_norm_sq * row_candidate_norm_sq);
        }
        row_cosine_min = std::min(row_cosine_min, row_cosine);
        if (row_rel > worst_row_rel) {
            worst_row_rel = row_rel;
            worst_row = row;
            worst_row_delta_norm = row_delta_norm;
            worst_row_ref_norm = row_ref_norm;
            worst_row_rmse = row_rmse;
            worst_row_max_abs = row_max_abs;
        }
    }

    double cosine = 0.0;
    if (norm_ref > 0.0 && norm_candidate > 0.0) {
        cosine = dot / std::sqrt(norm_ref * norm_candidate);
    }
    const double o_delta_norm = std::sqrt(sum_sq);
    const double o_ref_norm = std::sqrt(norm_ref);
    const double o_candidate_norm = std::sqrt(norm_candidate);
    const double o_delta_rel = o_delta_norm / std::max(o_ref_norm, 1.0e-12);

    std::cerr << "INFINILM_ATTENTION_LOCAL_CERT_DIFF"
              << " tag=" << tag
              << " step=" << step
              << " layer=" << layer_idx
              << " count=" << count
              << " rows=" << rows
              << " head_dim=" << head_dim
              << " real_len=" << real_total_seq_len
              << " total_len=" << total_seq_len
              << " unequal=" << unequal
              << " unequal_ratio=" << (count == 0 ? 0.0 : static_cast<double>(unequal) / count)
              << " mean_abs=" << (count == 0 ? 0.0 : sum_abs / count)
              << " max_abs=" << max_abs
              << " rmse=" << (count == 0 ? 0.0 : std::sqrt(sum_sq / count))
              << " cosine=" << cosine
              << " o_delta_norm=" << o_delta_norm
              << " o_ref_norm=" << o_ref_norm
              << " o_candidate_norm=" << o_candidate_norm
              << " o_delta_rel=" << o_delta_rel
              << " row_cosine_min=" << row_cosine_min
              << " worst_row=" << worst_row
              << " worst_row_delta_norm=" << worst_row_delta_norm
              << " worst_row_ref_norm=" << worst_row_ref_norm
              << " worst_row_delta_rel=" << worst_row_rel
              << " worst_row_rmse=" << worst_row_rmse
              << " worst_row_max_abs=" << worst_row_max_abs
              << std::endl;
}


bool infiniLmAttentionStateFusedV2bEnabled() {
    const char *env = std::getenv("INFINILM_KV_PAD_MODE");
    return env != nullptr && std::strcmp(env, "attention_state_fused_v2b") == 0;
}

size_t infiniLmAttentionStateFusedV2bSplitSize() {
    const char *env = std::getenv("INFINILM_ATTENTION_STATE_FUSED_V2B_SPLIT_SIZE");
    if (env == nullptr || env[0] == '\0') {
        return 512;
    }
    const auto value = static_cast<size_t>(std::strtoull(env, nullptr, 10));
    return value == 1024 ? 1024 : 512;
}


bool infiniLmAttentionStateFusedV2bLayerFallback(long layer_idx) {
    const char *env = std::getenv("INFINILM_ATTENTION_STATE_FUSED_V2B_FALLBACK_LAYERS");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    if (std::strcmp(env, "none") == 0 || std::strcmp(env, "0none") == 0) {
        return false;
    }
    if (std::strcmp(env, "all") == 0) {
        return true;
    }

    const char *p = env;
    while (*p != '\0') {
        while (*p == ',' || *p == ';' || *p == ' ') {
            ++p;
        }
        if (*p == '\0') {
            break;
        }
        char *end = nullptr;
        const long value = std::strtol(p, &end, 10);
        if (end != p && value == layer_idx) {
            return true;
        }
        if (end == p) {
            ++p;
        } else {
            p = end;
        }
    }
    return false;
}

using InfiniLmAttentionStateFusedDecodeV2bFn = int (*)(
    void *out,
    float *lse_out,
    const void *q,
    const void *k,
    const void *v,
    size_t kv_len,
    size_t batch_heads,
    size_t head_dim,
    ptrdiff_t q_stride,
    ptrdiff_t k_stride,
    ptrdiff_t v_stride,
    ptrdiff_t out_stride,
    float scale,
    size_t split_size);

InfiniLmAttentionStateFusedDecodeV2bFn infiniLmResolveAttentionStateFusedDecodeV2b() {
    constexpr const char *kSymbol = "infiniopAttentionStateFusedDecodeV2bBf16";
    static auto fn = reinterpret_cast<InfiniLmAttentionStateFusedDecodeV2bFn>(dlsym(RTLD_DEFAULT, kSymbol));
    if (fn == nullptr) {
        if (auto *handle = dlopen("libinfiniop.so", RTLD_LAZY | RTLD_GLOBAL)) {
            fn = reinterpret_cast<InfiniLmAttentionStateFusedDecodeV2bFn>(dlsym(handle, kSymbol));
        }
    }
    if (fn == nullptr) {
        if (auto *handle = dlopen("libinfinicore_cpp_api.so", RTLD_LAZY | RTLD_GLOBAL)) {
            fn = reinterpret_cast<InfiniLmAttentionStateFusedDecodeV2bFn>(dlsym(handle, kSymbol));
        }
    }
    return fn;
}


#ifndef INFINILM_V_CACHE_DECODE_LAYOUT_PATCH
#define INFINILM_V_CACHE_DECODE_LAYOUT_PATCH 1
#endif
#ifndef INFINILM_V_CACHE_DECODE_LAYOUT_SINGLE_STORAGE
#define INFINILM_V_CACHE_DECODE_LAYOUT_SINGLE_STORAGE 1
#endif
#ifndef INFINILM_V_CACHE_DECODE_LAYOUT_INPLACE_TRANSPOSE
#define INFINILM_V_CACHE_DECODE_LAYOUT_INPLACE_TRANSPOSE 1
#endif

bool infiniLmVCacheDecodeLayoutEnabled() {
    const char *mode = std::getenv("INFINILM_V_CACHE_DECODE_LAYOUT");
    return mode != nullptr &&
        (std::strcmp(mode, "page16_fp32_forward") == 0 ||
         std::strcmp(mode, "page16_fp32_warp") == 0 ||
         std::strcmp(mode, "page32_fp32_forward") == 0 ||
         std::strcmp(mode, "page16_fp32_reverse") == 0 ||
         std::strcmp(mode, "page32_fp32_reverse") == 0 ||
         std::strcmp(mode, "page16_bf16_forward") == 0 ||
         std::strcmp(mode, "page32_bf16_forward") == 0);
}

int infiniLmVCacheDecodeLayoutVariant() {
    const char *mode = std::getenv("INFINILM_V_CACHE_DECODE_LAYOUT");
    if (mode != nullptr && std::strcmp(mode, "page16_fp32_warp") == 0) {
        return 3;
    }
    if (mode != nullptr && std::strstr(mode, "_reverse") != nullptr) {
        return 1;
    }
    if (mode != nullptr && std::strstr(mode, "_bf16_") != nullptr) {
        return 2;
    }
    return 0;
}

using InfiniLmVCacheDecodeLayoutMatvecFn = int (*)(
    void *,
    const void *,
    const void *,
    size_t,
    size_t,
    size_t,
    size_t,
    size_t,
    ptrdiff_t,
    ptrdiff_t,
    ptrdiff_t,
    int);

InfiniLmVCacheDecodeLayoutMatvecFn
infiniLmResolveVCacheDecodeLayoutMatvec() {
    constexpr const char *symbol = "infiniopVCacheDecodeLayoutMatvecBf16";
    static auto fn = reinterpret_cast<InfiniLmVCacheDecodeLayoutMatvecFn>(
        dlsym(RTLD_DEFAULT, symbol));
    if (fn == nullptr) {
        if (auto *handle = dlopen("libinfiniop.so", RTLD_LAZY | RTLD_GLOBAL)) {
            fn = reinterpret_cast<InfiniLmVCacheDecodeLayoutMatvecFn>(
                dlsym(handle, symbol));
        }
    }
    return fn;
}

bool infiniLmAttentionStateFusedV2Enabled() {
    const char *env = std::getenv("INFINILM_KV_PAD_MODE");
    return env != nullptr && std::strcmp(env, "attention_state_fused_v2") == 0;
}

int infiniLmAttentionStateFusedV2Variant() {
    const char *env = std::getenv("INFINILM_ATTENTION_STATE_FUSED_V2_VARIANT");
    if (env != nullptr && std::strcmp(env, "block8") == 0) {
        return 1;
    }
    return 0;
}

using InfiniLmAttentionStateFusedDecodeV2Fn = int (*)(
    void *out,
    float *lse_out,
    const void *q,
    const void *k,
    const void *v,
    size_t kv_len,
    size_t batch_heads,
    size_t head_dim,
    ptrdiff_t q_stride,
    ptrdiff_t k_stride,
    ptrdiff_t v_stride,
    ptrdiff_t out_stride,
    float scale,
    int variant);

InfiniLmAttentionStateFusedDecodeV2Fn infiniLmResolveAttentionStateFusedDecodeV2() {
    constexpr const char *kSymbol = "infiniopAttentionStateFusedDecodeV2Bf16";
    static auto fn = reinterpret_cast<InfiniLmAttentionStateFusedDecodeV2Fn>(dlsym(RTLD_DEFAULT, kSymbol));
    if (fn == nullptr) {
        if (auto *handle = dlopen("libinfiniop.so", RTLD_LAZY | RTLD_GLOBAL)) {
            fn = reinterpret_cast<InfiniLmAttentionStateFusedDecodeV2Fn>(dlsym(handle, kSymbol));
        }
    }
    if (fn == nullptr) {
        if (auto *handle = dlopen("libinfinicore_cpp_api.so", RTLD_LAZY | RTLD_GLOBAL)) {
            fn = reinterpret_cast<InfiniLmAttentionStateFusedDecodeV2Fn>(dlsym(handle, kSymbol));
        }
    }
    return fn;
}

bool infiniLmAttentionStateFusedEnabled() {
    const char *env = std::getenv("INFINILM_KV_PAD_MODE");
    return env != nullptr && std::strcmp(env, "attention_state_fused") == 0;
}


using InfiniLmAttentionStateFusedDecodeFn = int (*)(
    void *out,
    float *lse_out,
    const void *q,
    const void *k,
    const void *v,
    size_t kv_len,
    size_t batch_heads,
    size_t head_dim,
    ptrdiff_t q_stride,
    ptrdiff_t k_stride,
    ptrdiff_t v_stride,
    ptrdiff_t out_stride,
    float scale);

InfiniLmAttentionStateFusedDecodeFn infiniLmResolveAttentionStateFusedDecode() {
    constexpr const char *kSymbol = "infiniopAttentionStateFusedDecodeBf16Prototype";
    static auto fn = reinterpret_cast<InfiniLmAttentionStateFusedDecodeFn>(dlsym(RTLD_DEFAULT, kSymbol));
    if (fn == nullptr) {
        if (auto *handle = dlopen("libinfiniop.so", RTLD_LAZY | RTLD_GLOBAL)) {
            fn = reinterpret_cast<InfiniLmAttentionStateFusedDecodeFn>(dlsym(handle, kSymbol));
        }
    }
    if (fn == nullptr) {
        if (auto *handle = dlopen("libinfinicore_cpp_api.so", RTLD_LAZY | RTLD_GLOBAL)) {
            fn = reinterpret_cast<InfiniLmAttentionStateFusedDecodeFn>(dlsym(handle, kSymbol));
        }
    }
    return fn;
}

size_t infiniLmKvPadVSplitMinLen() {
    const char *env = std::getenv("INFINILM_KV_PAD_V_SPLIT_MIN_LEN");
    if (env == nullptr || env[0] == '\0') {
        return 1024;
    }
    const int value = std::atoi(env);
    return value > 0 ? static_cast<size_t>(value) : 0;
}

size_t infiniLmKvPadQkSplitAlign() {
    const char *env = std::getenv("INFINILM_KV_PAD_ALIGN");
    if (env == nullptr || env[0] == '\0') {
        return 32;
    }
    size_t value = static_cast<size_t>(std::strtoull(env, nullptr, 10));
    return value > 1 ? value : 32;
}

bool infiniLmKvPadQkDiffTraceEnabled() {
    const char *env = std::getenv("INFINILM_KV_PAD_QK_DIFF_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

size_t infiniLmKvPadQkDiffLimit(const char *name, size_t fallback) {
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return fallback;
    }
    return static_cast<size_t>(std::strtoull(env, nullptr, 10));
}

float infiniLmTraceHalfToFloat(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    int32_t exp = static_cast<int32_t>((h >> 10) & 0x1fu);
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | (static_cast<uint32_t>(exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | (static_cast<uint32_t>(exp + 112) << 23) | (mant << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float infiniLmTraceValue(const std::byte *raw, infinicore::DataType dtype, size_t index) {
    if (dtype == infinicore::DataType::F32) {
        return reinterpret_cast<const float *>(raw)[index];
    }
    uint16_t bits16 = reinterpret_cast<const uint16_t *>(raw)[index];
    if (dtype == infinicore::DataType::BF16) {
        uint32_t bits32 = static_cast<uint32_t>(bits16) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits32, sizeof(value));
        return value;
    }
    return infiniLmTraceHalfToFloat(bits16);
}

uint64_t infiniLmTraceRawBits(const std::byte *raw, infinicore::DataType dtype, size_t index) {
    if (dtype == infinicore::DataType::F32) {
        uint32_t bits = 0;
        std::memcpy(&bits, raw + index * sizeof(uint32_t), sizeof(bits));
        return bits;
    }
    return reinterpret_cast<const uint16_t *>(raw)[index];
}

size_t infiniLmLayerwiseTraceStep(bool padded_mode, bool v_output, size_t layer_idx) {
    static thread_local std::vector<size_t> baseline_softmax;
    static thread_local std::vector<size_t> padded_softmax;
    static thread_local std::vector<size_t> baseline_vout;
    static thread_local std::vector<size_t> padded_vout;
    auto &counters = padded_mode
                         ? (v_output ? padded_vout : padded_softmax)
                         : (v_output ? baseline_vout : baseline_softmax);
    if (counters.size() <= layer_idx) {
        counters.resize(layer_idx + 1, 0);
    }
    return counters[layer_idx]++;
}

size_t infiniLmLayerwiseTraceLimit() {
    const char *env = std::getenv("INFINILM_KV_PAD_LAYER_TRACE_STEPS");
    if (env == nullptr || env[0] == '\0') {
        return 4;
    }
    return static_cast<size_t>(std::strtoull(env, nullptr, 10));
}

uint64_t infiniLmTracePrefixHash(const infinicore::Tensor &tensor,
                                 size_t rows,
                                 size_t physical_width,
                                 size_t real_width) {
    auto cpu = tensor->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::syncStream();
    const std::byte *raw = cpu->data();
    auto dtype = cpu->dtype();
    uint64_t hash = 1469598103934665603ULL;
    for (size_t row = 0; row < rows; ++row) {
        size_t base = row * physical_width;
        for (size_t col = 0; col < real_width; ++col) {
            hash ^= infiniLmTraceRawBits(raw, dtype, base + col);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

uint64_t infiniLmTraceTensorHash(const infinicore::Tensor &tensor) {
    auto cpu = tensor->contiguous()->to(infinicore::Device::cpu());
    infinicore::context::syncStream();
    const std::byte *raw = cpu->data();
    auto dtype = cpu->dtype();
    uint64_t hash = 1469598103934665603ULL;
    for (size_t index = 0; index < cpu->numel(); ++index) {
        hash ^= infiniLmTraceRawBits(raw, dtype, index);
        hash *= 1099511628211ULL;
    }
    return hash;
}

size_t firstI64TensorValue(const infinicore::Tensor &tensor, size_t fallback) {
    try {
        auto cpu_tensor = tensor->to(infinicore::Device::cpu());
        auto ptr = reinterpret_cast<const int64_t *>(cpu_tensor->data());
        if (ptr != nullptr && ptr[0] > 0) {
            return static_cast<size_t>(ptr[0]);
        }
    } catch (...) {
    }
    return fallback;
}
} // namespace

namespace infinilm::models::llama {

LlamaAttention::LlamaAttention(const LlamaConfig &config,
                               const infinicore::Device &device,
                               size_t layer_idx,
                               engine::distributed::RankInfo rank_info)
    : layer_idx_(layer_idx),
      hidden_size_(config.hidden_size),
      num_attention_heads_(config.num_attention_heads),
      num_key_value_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      kv_dim_(config.kv_dim()),
      use_bias_(config.attention_bias),
      use_output_bias_(config.attention_output_bias),
      use_qk_norm_(config.qk_norm),
      max_position_embeddings_(config.max_position_embeddings), rank_info_(rank_info) {
    const auto &dtype{config.dtype};

    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    int num_attention_heads = config.num_attention_heads;
    int num_key_value_heads = config.num_key_value_heads;

    if ((num_key_value_heads >= tp_size) && (0 == (num_key_value_heads % tp_size))) {
        this->num_attention_heads_ = num_attention_heads / tp_size;
        this->num_key_value_heads_ = num_key_value_heads / tp_size;
    } else {
        throw std::runtime_error("num_attention_heads / tp_size error.");
    }
    scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    // Initialize projection layers
    INFINILM_QKV_LINEAR_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, config.num_attention_heads, config.num_key_value_heads, use_bias_,
                             dtype, device, rank_info);
    // Output projection uses attention_output_bias (can be different from qkv)
    INFINICORE_NN_MODULE_INIT(o_proj, num_attention_heads * head_dim_, hidden_size_, use_output_bias_,
                              dtype, device, tp_rank, tp_size, rank_info.comm);

    // Initialize qk RMSNorm
    if (use_qk_norm_) {
        INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, config.rms_norm_eps, dtype, device);
        INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, config.rms_norm_eps, dtype, device);
    }
}

infinicore::Tensor LlamaAttention::forward_(const infinicore::Tensor &hidden_states,
                                            const infinicore::Tensor &position_ids,
                                            std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                            std::optional<infinicore::Tensor> past_sequence_lengths,
                                            std::optional<infinicore::Tensor> total_sequence_lengths) const {
    // Input shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    const auto phase = infinilm::profile::phase_from_tensor(hidden_states);
    INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "attention_total", hidden_states);

    // 1. Project Q, K, V
    auto [q, k, v] = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "attn_qkv_proj", hidden_states);
        return qkv_proj_->forward_split(hidden_states_mutable);
    }();

    if (use_qk_norm_) {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "attn_qk_norm", q);
        q = q_norm_->forward(q->view({batch_size * seq_len, num_attention_heads_, head_dim_}));
        k = k_norm_->forward(k->view({batch_size * seq_len, num_key_value_heads_, head_dim_}));
    }

    // 2. Reshape for multi-head attention
    // Reshape Q, K, V to include batch dimension
    // Python: query_states = self.q_proj(hidden_states).view(querys_shape)
    // The view operation requires the tensor to be contiguous in the required dimensions
    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    // 3. Prepare position_ids for RoPE - align with Python pattern
    // Python: bs, num = pos_ids.shape; pos_ids = pos_ids.view((bs * num,))
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }

    // 4. Apply RoPE to Q and K
    auto q_rope = infinicore::Tensor::empty({batch_size, num_attention_heads_, seq_len, head_dim_}, q_reshaped->dtype(), q_reshaped->device())->permute({0, 2, 1, 3});
    {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "attn_rope", q_reshaped);
        if (iv_rope_pair_model_enabled()) {
            rotary_emb_->forward_qk(q_rope, q_reshaped, k_reshaped, pos_ids_for_rope);
        } else {
        rotary_emb_->forward(q_rope, q_reshaped, pos_ids_for_rope); // [bs, seq_len, n_q_head, head_dim]
        rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);   // [bs, seq_len, n_kv_head, head_dim]
        }
    }

    // 5. Prepare KV caches
    // Convert to [batch, n_head, seq_len, head_dim] for cache
    // Ensure contiguous after permute for F16 compatibility with cache operations
    q_reshaped = q_rope->permute({0, 2, 1, 3});          // [bs, n_q_head, seq_len, head_dim]
    auto k_permuted = k_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]
    auto v_permuted = v_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]
    infinicore::Tensor k_total;                          // [bs, n_kv_head, total_seq_len, head_dim]
    infinicore::Tensor v_total;                          // [bs, n_kv_head, total_seq_len, head_dim]
    if (kv_cache == nullptr) {
        k_total = k_permuted;
        v_total = v_permuted;
    } else if (auto static_kv_cache = std::dynamic_pointer_cast<cache::StaticKVCache>(kv_cache)) {
        auto [k_total_tmp, v_total_tmp] = [&]() {
            INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "attn_cache_update", k_permuted);
            return static_kv_cache->update(layer_idx_, k_permuted, v_permuted, past_sequence_lengths.value());
        }();
        k_total = k_total_tmp;
        v_total = v_total_tmp;
    } else {
        throw std::runtime_error("LlamaAttention: Unsupported kvcache type");
    }
    auto total_seq_len = k_total->shape()[2];
    size_t real_total_seq_len = total_seq_len;
    if (total_sequence_lengths.has_value()) {
        real_total_seq_len = firstI64TensorValue(total_sequence_lengths.value(), total_seq_len);
    }
    if (real_total_seq_len > total_seq_len) {
        real_total_seq_len = total_seq_len;
    }


    const auto kv_cache_quant_config = infiniLmKvCacheQuantConfig();
    if (kv_cache_quant_config.enabled &&
        head_dim_ == 128 &&
        k_total->dtype() == infinicore::DataType::BF16) {
        const char *kv_quant_phase = seq_len == 1 ? "decode" : "prefill";
        if (kv_cache_quant_config.quantize_k) {
            k_total = infiniLmKvCacheMixedQuantizeDequantize(
                k_total,
                kv_cache_quant_config,
                static_cast<size_t>(layer_idx_),
                "K",
                kv_quant_phase,
                batch_size,
                num_key_value_heads_,
                total_seq_len,
                real_total_seq_len,
                head_dim_);
        }
        if (kv_cache_quant_config.quantize_v) {
            v_total = infiniLmKvCacheMixedQuantizeDequantize(
                v_total,
                kv_cache_quant_config,
                static_cast<size_t>(layer_idx_),
                "V",
                kv_quant_phase,
                batch_size,
                num_key_value_heads_,
                total_seq_len,
                real_total_seq_len,
                head_dim_);
        }
        if (layer_idx_ == 0 && seq_len == 1) {
            std::cerr << "INFINILM_KV_CACHE_MIXED_QUANT_DISPATCH"
                      << " mode=" << kv_cache_quant_config.mode
                      << " bits=" << kv_cache_quant_config.bits
                      << " group_size=" << kv_cache_quant_config.group_size
                      << " residual_tokens=" << kv_cache_quant_config.residual_tokens
                      << " batch=" << batch_size
                      << " real_len=" << real_total_seq_len
                      << " total_len=" << total_seq_len
                      << " head_dim=" << head_dim_
                      << std::endl;
        }
    }

    // 6. Compute attention
    size_t ngroup = num_attention_heads_ / num_key_value_heads_;
    auto Q = q_reshaped->view({batch_size * num_key_value_heads_, ngroup * seq_len, head_dim_});
    auto K = k_total->view({batch_size * num_key_value_heads_, total_seq_len, head_dim_});
    auto static_decode_cache =
        std::dynamic_pointer_cast<cache::StaticKVCache>(kv_cache);
    const bool use_v_cache_decode_layout =
        infiniLmVCacheDecodeLayoutEnabled() &&
        seq_len == 1 &&
        ngroup == 1 &&
        head_dim_ == 128 &&
        real_total_seq_len > 0 &&
        static_decode_cache != nullptr &&
        static_decode_cache->has_v_decode_layout(layer_idx_);
    infinicore::Tensor V;
    if (!use_v_cache_decode_layout) {
        V = v_total->view(
            {batch_size * num_key_value_heads_,
             total_seq_len,
             head_dim_});
    }


    // Diagnostic stage isolation for identical-history batches. qk_only keeps
    // the full-batch V GEMM, v_only keeps full-batch QK/softmax, and qk_v
    // reduces both stages. These modes are evidence-only and never selected by
    // the production policy.
    if (infiniLmSharedPrefixIsolationEnabled() &&
        infiniLmSharedPrefixIdenticalBatchContract() &&
        seq_len == 1 &&
        batch_size >= infiniLmSharedPrefixKvReuseMinBatch() &&
        real_total_seq_len >= infiniLmSharedPrefixKvReuseMinKv() &&
        ngroup == 1 &&
        Q->dtype() == infinicore::DataType::BF16) {
        const bool reuse_qk = infiniLmSharedPrefixIsolationReuseQk();
        const bool reuse_v = infiniLmSharedPrefixIsolationReuseV();
        const size_t heads_per_batch = num_key_value_heads_;
        const size_t weight_rows = reuse_qk
            ? num_attention_heads_
            : batch_size * num_attention_heads_;

        INFINILM_PROFILE_SCOPE(
            phase,
            static_cast<long>(layer_idx_),
            rank_info_.tp_rank,
            "attn_shared_prefix_isolation_total",
            Q);

        auto Q_work = reuse_qk
            ? Q->narrow({{0, 0, heads_per_batch}})
            : Q;
        auto K_work = reuse_qk
            ? K->narrow({{0, 0, heads_per_batch}})
            : K;
        auto K_work_transposed = K_work->permute({0, 2, 1});

        auto work_weight = [&]() {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_shared_prefix_isolation_qk",
                Q_work);
            return infinicore::op::matmul(Q_work, K_work_transposed, scaling_);
        }();

        auto work_softmax = work_weight->view(
            {weight_rows, seq_len, total_seq_len});
        {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_shared_prefix_isolation_softmax",
                work_softmax);
            if (real_total_seq_len < total_seq_len) {
                infiniopSetCausalSoftmaxEffectiveLength(real_total_seq_len);
            }
            infinicore::op::causal_softmax_(work_softmax, work_softmax);
            if (real_total_seq_len < total_seq_len) {
                infiniopSetCausalSoftmaxEffectiveLength(0);
            }
        }

        auto weight_for_v = work_weight;
        auto V_for_matmul = V;
        uint64_t score_copy_bytes = 0;
        if (reuse_qk && !reuse_v) {
            auto full_weight = infinicore::Tensor::empty(
                {batch_size * num_attention_heads_, seq_len, total_seq_len},
                Q->dtype(),
                Q->device());
            {
                INFINILM_PROFILE_SCOPE(
                    phase,
                    static_cast<long>(layer_idx_),
                    rank_info_.tp_rank,
                    "attn_shared_prefix_isolation_score_copy",
                    full_weight);
                for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                    full_weight
                        ->narrow(
                            {{0,
                              batch_idx * num_attention_heads_,
                              num_attention_heads_}})
                        ->copy_from(work_weight);
                }
            }
            weight_for_v = full_weight;
            score_copy_bytes =
                static_cast<uint64_t>(batch_size) *
                static_cast<uint64_t>(num_attention_heads_) *
                static_cast<uint64_t>(seq_len) *
                static_cast<uint64_t>(total_seq_len) * 2;
        } else if (!reuse_qk && reuse_v) {
            weight_for_v = work_weight->narrow({{0, 0, heads_per_batch}});
            V_for_matmul = V->narrow({{0, 0, heads_per_batch}});
        } else if (reuse_qk && reuse_v) {
            V_for_matmul = V->narrow({{0, 0, heads_per_batch}});
        }

        auto work_out = [&]() {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_shared_prefix_isolation_v",
                weight_for_v);
            return infinicore::op::matmul(weight_for_v, V_for_matmul);
        }();

        infinicore::Tensor out = work_out;
        uint64_t output_copy_bytes = 0;
        if (reuse_v) {
            out = infinicore::Tensor::empty(
                {batch_size * num_attention_heads_, seq_len, head_dim_},
                Q->dtype(),
                Q->device());
            {
                INFINILM_PROFILE_SCOPE(
                    phase,
                    static_cast<long>(layer_idx_),
                    rank_info_.tp_rank,
                    "attn_shared_prefix_isolation_output_copy",
                    out);
                for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                    out->narrow(
                           {{0,
                             batch_idx * num_attention_heads_,
                             num_attention_heads_}})
                        ->copy_from(work_out);
                }
            }
            output_copy_bytes =
                static_cast<uint64_t>(batch_size) *
                static_cast<uint64_t>(num_attention_heads_) *
                static_cast<uint64_t>(seq_len) *
                static_cast<uint64_t>(head_dim_) * 2;
        }

        if (infiniLmSharedPrefixKvReuseTraceEnabled() && layer_idx_ == 0) {
            const uint64_t one_stage_bytes =
                static_cast<uint64_t>(num_key_value_heads_) *
                static_cast<uint64_t>(real_total_seq_len) *
                static_cast<uint64_t>(head_dim_) * 2;
            const uint64_t baseline_kv_bytes =
                static_cast<uint64_t>(batch_size) * one_stage_bytes * 2;
            const uint64_t candidate_k_bytes = reuse_qk
                ? one_stage_bytes
                : static_cast<uint64_t>(batch_size) * one_stage_bytes;
            const uint64_t candidate_v_bytes = reuse_v
                ? one_stage_bytes
                : static_cast<uint64_t>(batch_size) * one_stage_bytes;
            const uint64_t candidate_kv_bytes =
                candidate_k_bytes + candidate_v_bytes;
            std::cerr << "INFINILM_SHARED_PREFIX_KV_REUSE_ISOLATION"
                      << " mode=" << infiniLmSharedPrefixIsolationMode()
                      << " batch=" << batch_size
                      << " kv_len=" << real_total_seq_len
                      << " baseline_kv_bytes=" << baseline_kv_bytes
                      << " candidate_kv_bytes=" << candidate_kv_bytes
                      << " score_copy_bytes=" << score_copy_bytes
                      << " output_copy_bytes=" << output_copy_bytes
                      << " ideal_kv_read_speedup="
                      << static_cast<double>(baseline_kv_bytes) /
                             static_cast<double>(candidate_kv_bytes)
                      << std::endl;
        }

        auto attn_output = out->view(
                                   {batch_size, num_attention_heads_, seq_len, head_dim_})
                               ->permute({0, 2, 1, 3})
                               ->contiguous()
                               ->view(
                                   {batch_size,
                                    seq_len,
                                    num_attention_heads_ * head_dim_});
        auto output = [&]() {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_o_proj",
                attn_output);
            return o_proj_->forward(attn_output);
        }();
        return output;
    }


    // Exact first-stage shared-prefix prototype. The scheduler/test harness
    // must guarantee that every batch row has the same complete token history.
    // Partial prefix sharing with divergent suffixes is intentionally excluded.
    if (infiniLmSharedPrefixKvReuseEnabled() &&
        infiniLmSharedPrefixIdenticalBatchContract() &&
        seq_len == 1 &&
        batch_size >= infiniLmSharedPrefixKvReuseMinBatch() &&
        real_total_seq_len >= infiniLmSharedPrefixKvReuseMinKv() &&
        ngroup == 1 &&
        Q->dtype() == infinicore::DataType::BF16) {
        INFINILM_PROFILE_SCOPE(
            phase,
            static_cast<long>(layer_idx_),
            rank_info_.tp_rank,
            "attn_shared_prefix_total",
            Q);

        const size_t heads_per_batch = num_key_value_heads_;
        auto Q_shared = Q->narrow({{0, 0, heads_per_batch}});
        auto K_shared = K->narrow({{0, 0, heads_per_batch}});
        auto V_shared = V->narrow({{0, 0, heads_per_batch}});
        auto K_shared_transposed = K_shared->permute({0, 2, 1});

        auto shared_weight = [&]() {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_shared_prefix_qk_matmul",
                Q_shared);
            return infinicore::op::matmul(Q_shared, K_shared_transposed, scaling_);
        }();

        auto shared_weight_softmax = shared_weight->view(
            {num_attention_heads_, seq_len, total_seq_len});
        {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_shared_prefix_causal_softmax",
                shared_weight_softmax);
            if (real_total_seq_len < total_seq_len) {
                infiniopSetCausalSoftmaxEffectiveLength(real_total_seq_len);
            }
            infinicore::op::causal_softmax_(
                shared_weight_softmax, shared_weight_softmax);
            if (real_total_seq_len < total_seq_len) {
                infiniopSetCausalSoftmaxEffectiveLength(0);
            }
        }

        auto shared_out = [&]() {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_shared_prefix_v_matmul",
                shared_weight);
            return infinicore::op::matmul(shared_weight, V_shared);
        }();

        auto out = infinicore::Tensor::empty(
            {batch_size * num_attention_heads_, seq_len, head_dim_},
            Q->dtype(),
            Q->device());
        {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_shared_prefix_output_copy",
                out);
            for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                out->narrow({{0, batch_idx * num_attention_heads_, num_attention_heads_}})
                    ->copy_from(shared_out);
            }
        }

        if (infiniLmSharedPrefixKvReuseTraceEnabled() && layer_idx_ == 0) {
            const uint64_t bytes_per_value = 2;
            const uint64_t one_batch_kv_bytes =
                static_cast<uint64_t>(num_key_value_heads_) *
                static_cast<uint64_t>(real_total_seq_len) *
                static_cast<uint64_t>(head_dim_) * bytes_per_value * 2;
            const uint64_t baseline_kv_bytes =
                static_cast<uint64_t>(batch_size) * one_batch_kv_bytes;
            const uint64_t output_copy_bytes =
                static_cast<uint64_t>(batch_size) *
                static_cast<uint64_t>(num_attention_heads_) *
                static_cast<uint64_t>(seq_len) *
                static_cast<uint64_t>(head_dim_) * bytes_per_value;
            std::cerr << "INFINILM_SHARED_PREFIX_KV_REUSE"
                      << " mode=identical_batch_contract"
                      << " batch=" << batch_size
                      << " kv_len=" << real_total_seq_len
                      << " heads=" << num_key_value_heads_
                      << " head_dim=" << head_dim_
                      << " baseline_kv_bytes=" << baseline_kv_bytes
                      << " candidate_kv_bytes=" << one_batch_kv_bytes
                      << " output_copy_bytes=" << output_copy_bytes
                      << " ideal_kv_read_speedup="
                      << static_cast<double>(baseline_kv_bytes) /
                             static_cast<double>(one_batch_kv_bytes)
                      << std::endl;
        }

        auto attn_output = out->view(
                                   {batch_size, num_attention_heads_, seq_len, head_dim_})
                               ->permute({0, 2, 1, 3})
                               ->contiguous()
                               ->view(
                                   {batch_size,
                                    seq_len,
                                    num_attention_heads_ * head_dim_});
        auto output = [&]() {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_o_proj",
                attn_output);
            return o_proj_->forward(attn_output);
        }();
        return output;
    }


    if (infiniLmAttentionStateFusedV2bEnabled() &&
        !infiniLmAttentionStateFusedV2bLayerFallback(static_cast<long>(layer_idx_)) &&
        seq_len == 1 &&
        ngroup == 1 &&
        head_dim_ == 128 &&
        real_total_seq_len > 0 &&
        Q->dtype() == infinicore::DataType::BF16) {
        auto Q_fused = Q->contiguous();
        auto K_fused = K;
        auto V_fused = V;
        auto out = infinicore::Tensor::empty(
            {batch_size * num_attention_heads_, seq_len, head_dim_},
            Q->dtype(),
            Q->device());

        {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_state_fused_decode_v2b",
                Q_fused);
            auto fused_decode_fn = infiniLmResolveAttentionStateFusedDecodeV2b();
            if (fused_decode_fn == nullptr) {
                throw std::runtime_error("attention_state_fused_decode_v2b symbol is not available");
            }
            const size_t split_size = infiniLmAttentionStateFusedV2bSplitSize();
            const char *attr_trace_env = std::getenv("INFINICORE_ATTENTION_STATE_FUSED_V2B_ATTR_TRACE");
            if (attr_trace_env != nullptr && attr_trace_env[0] != '\0' && attr_trace_env[0] != '0') {
                char layer_buf[32];
                char heads_buf[32];
                std::snprintf(layer_buf, sizeof(layer_buf), "%ld", static_cast<long>(layer_idx_));
                std::snprintf(heads_buf, sizeof(heads_buf), "%zu", static_cast<size_t>(num_attention_heads_));
                setenv("INFINILM_ATTENTION_STATE_FUSED_V2B_TRACE_LAYER", layer_buf, 1);
                setenv("INFINILM_ATTENTION_STATE_FUSED_V2B_TRACE_HEADS", heads_buf, 1);
            }

            int status = fused_decode_fn(
                out->data(),
                nullptr,
                Q_fused->data(),
                K_fused->data(),
                V_fused->data(),
                real_total_seq_len,
                batch_size * num_attention_heads_,
                head_dim_,
                seq_len * head_dim_,
                K_fused->stride(0),
                V_fused->stride(0),
                seq_len * head_dim_,
                scaling_,
                split_size);
            if (status != 0) {
                throw std::runtime_error("attention_state_fused_decode_v2b backend failed");
            }
        }

        if (layer_idx_ == 0) {
            std::cerr << "INFINILM_ATTENTION_STATE_FUSED_DECODE_V2B real_len="
                      << real_total_seq_len
                      << " total_len=" << total_seq_len
                      << " batch_heads=" << batch_size * num_attention_heads_
                      << " head_dim=" << head_dim_
                      << " split_size=" << infiniLmAttentionStateFusedV2bSplitSize()
                      << std::endl;
        }

        auto attn_output = out->view({batch_size, num_attention_heads_, seq_len, head_dim_})
                               ->permute({0, 2, 1, 3})
                               ->contiguous()
                               ->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
        auto output = [&]() {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_o_proj",
                attn_output);
            return o_proj_->forward(attn_output);
        }();
        return output;
    }



    if (infiniLmAttentionStateFusedV2Enabled() &&
        seq_len == 1 &&
        ngroup == 1 &&
        head_dim_ == 128 &&
        real_total_seq_len > 0 &&
        Q->dtype() == infinicore::DataType::BF16) {
        auto Q_fused = Q->contiguous();
        auto K_fused = K;
        auto V_fused = V;
        auto out = infinicore::Tensor::empty(
            {batch_size * num_attention_heads_, seq_len, head_dim_},
            Q->dtype(),
            Q->device());

        {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_state_fused_decode_v2",
                Q_fused);
            auto fused_decode_fn = infiniLmResolveAttentionStateFusedDecodeV2();
            if (fused_decode_fn == nullptr) {
                throw std::runtime_error("attention_state_fused_decode_v2 symbol is not available");
            }
            const int variant = infiniLmAttentionStateFusedV2Variant();
            int status = fused_decode_fn(
                out->data(),
                nullptr,
                Q_fused->data(),
                K_fused->data(),
                V_fused->data(),
                real_total_seq_len,
                batch_size * num_attention_heads_,
                head_dim_,
                seq_len * head_dim_,
                K_fused->stride(0),
                V_fused->stride(0),
                seq_len * head_dim_,
                scaling_,
                variant);
            if (status != 0) {
                throw std::runtime_error("attention_state_fused_decode_v2 backend failed");
            }
            infiniLmAttentionLocalCertTrace(
                out,
                layer_idx_,
                real_total_seq_len,
                total_seq_len,
                batch_size * num_attention_heads_,
                head_dim_);

        }

        if (layer_idx_ == 0) {
            std::cerr << "INFINILM_ATTENTION_STATE_FUSED_DECODE_V2 real_len="
                      << real_total_seq_len
                      << " total_len=" << total_seq_len
                      << " batch_heads=" << batch_size * num_attention_heads_
                      << " head_dim=" << head_dim_
                      << " variant=" << infiniLmAttentionStateFusedV2Variant()
                      << std::endl;
        }

        auto attn_output = out->view({batch_size, num_attention_heads_, seq_len, head_dim_})
                               ->permute({0, 2, 1, 3})
                               ->contiguous()
                               ->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
        auto output = [&]() {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_o_proj",
                attn_output);
            return o_proj_->forward(attn_output);
        }();
        return output;
    }



    if (infiniLmAttentionStateFusedEnabled() &&
        seq_len == 1 &&
        ngroup == 1 &&
        head_dim_ == 128 &&
        real_total_seq_len > 0 &&
        Q->dtype() == infinicore::DataType::BF16) {
        auto Q_fused = Q->contiguous();
        auto K_fused = K;
        auto V_fused = V;
        auto out = infinicore::Tensor::empty(
            {batch_size * num_attention_heads_, seq_len, head_dim_},
            Q->dtype(),
            Q->device());

        {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_state_fused_decode",
                Q_fused);
            auto fused_decode_fn = infiniLmResolveAttentionStateFusedDecode();
            if (fused_decode_fn == nullptr) {
                throw std::runtime_error("attention_state_fused_decode symbol is not available");
            }
            int status = fused_decode_fn(
                out->data(),
                nullptr,
                Q_fused->data(),
                K_fused->data(),
                V_fused->data(),
                real_total_seq_len,
                batch_size * num_attention_heads_,
                head_dim_,
                seq_len * head_dim_,
                K_fused->stride(0),
                V_fused->stride(0),
                seq_len * head_dim_,
                scaling_);
            if (status != 0) {
                throw std::runtime_error("attention_state_fused_decode backend prototype failed");
            }
        }

        if (layer_idx_ == 0) {
            std::cerr << "INFINILM_ATTENTION_STATE_FUSED_DECODE real_len="
                      << real_total_seq_len
                      << " total_len=" << total_seq_len
                      << " batch_heads=" << batch_size * num_attention_heads_
                      << " head_dim=" << head_dim_
                      << std::endl;
        }

        auto attn_output = out->view({batch_size, num_attention_heads_, seq_len, head_dim_})
                               ->permute({0, 2, 1, 3})
                               ->contiguous()
                               ->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
        auto output = [&]() {
            INFINILM_PROFILE_SCOPE(
                phase,
                static_cast<long>(layer_idx_),
                rank_info_.tp_rank,
                "attn_o_proj",
                attn_output);
            return o_proj_->forward(attn_output);
        }();
        return output;
    }

    auto K_transposed = K->permute({0, 2, 1}); // [bs * n_kv_head, head_dim, total_seq_len]

    auto attn_weight = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "attn_qk_matmul", Q);
        if (infiniLmKvPadQkFixedEnabled() && seq_len == 1 &&
            real_total_seq_len < total_seq_len) {
            const auto q_shape = Q->shape();
            auto K_logical = K_transposed->narrow({{2, 0, real_total_seq_len}});
            auto logical_weight = infinicore::op::matmul(Q, K_logical, scaling_);
            auto padded_weight = infinicore::Tensor::zeros(
                {q_shape[0], q_shape[1], total_seq_len},
                Q->dtype(), Q->device());
            padded_weight->narrow({{2, 0, real_total_seq_len}})
                ->copy_from(logical_weight);
            if (layer_idx_ == 0) {
                std::cerr << "INFINILM_KV_PAD_QK_FIXED real_len="
                          << real_total_seq_len
                          << " physical_len=" << total_seq_len
                          << std::endl;
            }
            return padded_weight;
        }

        if (infiniLmKvPadQkSplitEnabled() && seq_len == 1 &&
            real_total_seq_len < total_seq_len) {
            const size_t align = infiniLmKvPadQkSplitAlign();
            const size_t prefix_len = (real_total_seq_len / align) * align;
            const size_t tail_len = real_total_seq_len - prefix_len;
            const auto q_shape = Q->shape();
            auto split_weight = infinicore::Tensor::zeros(
                {q_shape[0], q_shape[1], total_seq_len},
                Q->dtype(), Q->device());

            if (prefix_len > 0) {
                auto K_prefix = K_transposed->narrow({{2, 0, prefix_len}});
                auto prefix_weight = infinicore::op::matmul(Q, K_prefix, scaling_);
                split_weight->narrow({{2, 0, prefix_len}})->copy_from(prefix_weight);
            }
            if (tail_len > 0) {
                auto K_tail = K_transposed->narrow({{2, prefix_len, tail_len}});
                auto tail_weight = infinicore::op::matmul(Q, K_tail, scaling_);
                split_weight->narrow({{2, prefix_len, tail_len}})->copy_from(tail_weight);
            }
            if (layer_idx_ == 0) {
                std::cerr << "INFINILM_KV_PAD_QK_SPLIT real_len="
                          << real_total_seq_len
                          << " physical_len=" << total_seq_len
                          << " prefix_len=" << prefix_len
                          << " tail_len=" << tail_len
                          << " align=" << align
                          << std::endl;
            }
            return split_weight;
        }
        return infinicore::op::matmul(Q, K_transposed, scaling_); // [bs * n_kv_head, ng * seq_len, total_seq_len]
    }();


    if (infiniLmKvPadQkDiffTraceEnabled() && seq_len == 1 &&
        layer_idx_ < infiniLmKvPadQkDiffLimit("INFINILM_KV_PAD_QK_DIFF_LAYERS", 2)) {
        static thread_local std::vector<size_t> baseline_steps;
        static thread_local std::vector<size_t> padded_steps;
        static thread_local std::unordered_map<uint64_t, std::vector<float>> baseline_scores;
        static thread_local std::unordered_map<uint64_t, uint64_t> baseline_hashes;
        bool padded_mode = real_total_seq_len < total_seq_len;
        auto &steps = padded_mode ? padded_steps : baseline_steps;
        if (steps.size() <= layer_idx_) {
            steps.resize(layer_idx_ + 1, 0);
        }
        size_t trace_step = steps[layer_idx_]++;
        if (trace_step < infiniLmKvPadQkDiffLimit("INFINILM_KV_PAD_QK_DIFF_STEPS", 2)) {
            auto score_cpu = attn_weight->contiguous()->to(infinicore::Device::cpu());
            infinicore::context::syncStream();
            const std::byte *raw = score_cpu->data();
            auto dtype = score_cpu->dtype();
            size_t rows = score_cpu->numel() / total_seq_len;
            std::vector<float> real_scores;
            real_scores.reserve(rows * real_total_seq_len);
            uint64_t hash = 1469598103934665603ULL;
            for (size_t row = 0; row < rows; ++row) {
                size_t base = row * total_seq_len;
                for (size_t col = 0; col < real_total_seq_len; ++col) {
                    size_t index = base + col;
                    real_scores.push_back(infiniLmTraceValue(raw, dtype, index));
                    hash ^= infiniLmTraceRawBits(raw, dtype, index);
                    hash *= 1099511628211ULL;
                }
            }
            uint64_t key = (static_cast<uint64_t>(layer_idx_) << 32) | trace_step;
            if (!padded_mode) {
                baseline_scores[key] = std::move(real_scores);
                baseline_hashes[key] = hash;
                std::cerr << "INFINILM_KV_PAD_QK_BASELINE layer=" << layer_idx_
                          << " step=" << trace_step
                          << " count=" << baseline_scores[key].size()
                          << " hash=" << hash
                          << std::endl;
            } else {
                auto it = baseline_scores.find(key);
                if (it == baseline_scores.end() || it->second.size() != real_scores.size()) {
                    std::cerr << "INFINILM_KV_PAD_QK_DIFF_MISSING layer=" << layer_idx_
                              << " step=" << trace_step
                              << " padded_count=" << real_scores.size()
                              << std::endl;
                } else {
                    double sum_abs = 0.0;
                    double max_abs = 0.0;
                    double max_rel = 0.0;
                    size_t unequal = 0;
                    for (size_t i = 0; i < real_scores.size(); ++i) {
                        double lhs = static_cast<double>(it->second[i]);
                        double rhs = static_cast<double>(real_scores[i]);
                        double abs_diff = std::abs(lhs - rhs);
                        if (abs_diff != 0.0) {
                            ++unequal;
                        }
                        sum_abs += abs_diff;
                        max_abs = std::max(max_abs, abs_diff);
                        max_rel = std::max(max_rel, abs_diff / std::max(std::abs(lhs), 1.0e-12));
                    }
                    std::cerr << "INFINILM_KV_PAD_QK_DIFF layer=" << layer_idx_
                              << " step=" << trace_step
                              << " count=" << real_scores.size()
                              << " unequal=" << unequal
                              << " unequal_ratio=" << (static_cast<double>(unequal) / real_scores.size())
                              << " mean_abs=" << (sum_abs / real_scores.size())
                              << " max_abs=" << max_abs
                              << " max_rel=" << max_rel
                              << " baseline_hash=" << baseline_hashes[key]
                              << " padded_hash=" << hash
                              << std::endl;
                }
            }
        }
    }

    auto attn_weight_softmax = attn_weight->view({batch_size * num_attention_heads_, seq_len, total_seq_len});
    {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "attn_causal_softmax", attn_weight_softmax);
        if (real_total_seq_len < total_seq_len) {
        infiniopSetCausalSoftmaxEffectiveLength(real_total_seq_len);
        if (infiniLmKvPadTraceEnabled() && layer_idx_ == 0 && seq_len == 1) {
            std::cerr << "INFINILM_KV_PAD_ATTN_TRACE layer=" << layer_idx_
                      << " seq_len=" << seq_len
                      << " total=" << total_seq_len
                      << " real=" << real_total_seq_len
                      << " effective_get=" << infiniopGetCausalSoftmaxEffectiveLength()
                      << std::endl;
        }
    }
    infinicore::op::causal_softmax_(attn_weight_softmax, attn_weight_softmax);
    if (infiniLmKvBlockMultiRepTraceEnabled() &&
        seq_len == 1 && ngroup == 1 && head_dim_ == 128) {
        infiniLmKvBlockMultiRepTrace(
            Q,
            K,
            attn_weight_softmax,
            static_cast<size_t>(layer_idx_),
            real_total_seq_len,
            total_seq_len,
            batch_size,
            num_attention_heads_,
            head_dim_,
            scaling_);
    }
    if (infiniLmKvBlockMeanKeyTraceEnabled() &&
        seq_len == 1 && ngroup == 1 && head_dim_ == 128) {
        infiniLmKvBlockMeanKeyTrace(
            Q,
            K,
            attn_weight_softmax,
            static_cast<size_t>(layer_idx_),
            real_total_seq_len,
            total_seq_len,
            batch_size,
            num_attention_heads_,
            head_dim_);
    }
    if (infiniLmKvBlockMassTraceEnabled() && seq_len == 1) {
        infiniLmKvBlockMassTrace(
            attn_weight_softmax,
            static_cast<size_t>(layer_idx_),
            real_total_seq_len,
            total_seq_len,
            batch_size * num_attention_heads_ * seq_len);
    }
    if (infiniLmKvBlockHeadTraceEnabled() && seq_len == 1) {
        infiniLmKvBlockHeadTrace(
            attn_weight_softmax,
            static_cast<size_t>(layer_idx_),
            real_total_seq_len,
            total_seq_len,
            batch_size,
            num_attention_heads_);
    }
    if (infiniLmKvPadNumericTraceEnabled() && layer_idx_ == 0 && seq_len == 1) {
        static thread_local size_t baseline_softmax_step = 0;
        static thread_local size_t padded_softmax_step = 0;
        bool padded_mode = real_total_seq_len < total_seq_len;
        size_t trace_step = padded_mode ? padded_softmax_step++ : baseline_softmax_step++;
        auto trace_cpu = attn_weight_softmax->contiguous()->to(infinicore::Device::cpu());
        infinicore::context::syncStream();
        const std::byte *raw = trace_cpu->data();
        auto dtype = trace_cpu->dtype();
        size_t rows = batch_size * num_attention_heads_ * seq_len;
        double real_sum = 0.0;
        double padded_abs_sum = 0.0;
        double padded_max_abs = 0.0;
        double sumsq = 0.0;
        uint64_t hash = 1469598103934665603ULL;
        for (size_t row = 0; row < rows; ++row) {
            size_t base = row * total_seq_len;
            for (size_t col = 0; col < real_total_seq_len; ++col) {
                size_t index = base + col;
                float value = infiniLmTraceValue(raw, dtype, index);
                real_sum += static_cast<double>(value);
                sumsq += static_cast<double>(value) * static_cast<double>(value);
                hash ^= infiniLmTraceRawBits(raw, dtype, index);
                hash *= 1099511628211ULL;
            }
            for (size_t col = real_total_seq_len; col < total_seq_len; ++col) {
                float value = std::abs(infiniLmTraceValue(raw, dtype, base + col));
                padded_abs_sum += static_cast<double>(value);
                padded_max_abs = std::max(padded_max_abs, static_cast<double>(value));
            }
        }
        std::cerr << "INFINILM_KV_PAD_SOFTMAX_NUMERIC mode=" << (padded_mode ? "padded" : "baseline")
                  << " step=" << trace_step
                  << " total=" << total_seq_len
                  << " real=" << real_total_seq_len
                  << " rows=" << rows
                  << " real_sum=" << real_sum
                  << " padded_abs_sum=" << padded_abs_sum
                  << " padded_max_abs=" << padded_max_abs
                  << " sumsq=" << sumsq
                  << " hash=" << hash
                  << std::endl;
    }
    if (infiniLmKvPadNumericTraceEnabled() && seq_len == 1) {
        bool padded_mode = real_total_seq_len < total_seq_len;
        size_t trace_step = infiniLmLayerwiseTraceStep(padded_mode, false, layer_idx_);
        if (trace_step < infiniLmLayerwiseTraceLimit()) {
            size_t rows = batch_size * num_attention_heads_ * seq_len;
            uint64_t layer_hash = infiniLmTracePrefixHash(
                attn_weight_softmax, rows, total_seq_len, real_total_seq_len);
            std::cerr << "INFINILM_KV_PAD_LAYER_SOFTMAX mode=" << (padded_mode ? "padded" : "baseline")
                      << " layer=" << layer_idx_
                      << " step=" << trace_step
                      << " total=" << total_seq_len
                      << " real=" << real_total_seq_len
                      << " hash=" << layer_hash
                      << std::endl;
        }
    }
    if (real_total_seq_len < total_seq_len) {
        if (infiniLmKvPadTraceEnabled() && layer_idx_ == 0 && seq_len == 1) {
            std::cerr << "INFINILM_KV_PAD_ATTN_AFTER layer=" << layer_idx_
                      << " total=" << total_seq_len
                      << " real=" << real_total_seq_len
                      << " effective_get=" << infiniopGetCausalSoftmaxEffectiveLength()
                      << std::endl;
        }
        infiniopSetCausalSoftmaxEffectiveLength(0);
    }
    }

    auto attn_weight_for_v = attn_weight;
    auto V_for_matmul = V;
    const bool v_split_enabled =
        infiniLmKvPadVSplitEnabled() && seq_len == 1 &&
        real_total_seq_len >= infiniLmKvPadVSplitMinLen() &&
        real_total_seq_len < total_seq_len;
    if (!use_v_cache_decode_layout &&
        infiniLmKvPadQkOnlyEnabled() &&
        real_total_seq_len < total_seq_len) {
        attn_weight_for_v = attn_weight->narrow({{2, 0, real_total_seq_len}});
        V_for_matmul = V->narrow({{1, 0, real_total_seq_len}});
        if (layer_idx_ == 0 && seq_len == 1) {
            std::cerr << "INFINILM_KV_PAD_QK_ONLY qk_len=" << total_seq_len
                      << " v_len=" << real_total_seq_len
                      << std::endl;
        }
    }

    // INFINILM_V_LEGACY_ORDER_CAPTURE_PATCH
    const char *v_contract_capture_dir =
        std::getenv("INFINILM_V_LEGACY_CONTRACT_CAPTURE_DIR");
    const char *v_contract_layout_mode =
        std::getenv("INFINILM_V_CACHE_DECODE_LAYOUT");
    static thread_local bool v_contract_capture_done = false;
    if (!v_contract_capture_done &&
        v_contract_capture_dir != nullptr &&
        v_contract_capture_dir[0] != '\0' &&
        (v_contract_layout_mode == nullptr ||
         v_contract_layout_mode[0] == '\0' ||
         v_contract_layout_mode[0] == '0') &&
        layer_idx_ == 0 &&
        seq_len == 1 &&
        real_total_seq_len >= 1024 &&
        attn_weight_for_v->dtype() == infinicore::DataType::BF16 &&
        V->dtype() == infinicore::DataType::BF16) {
        auto capture_weights =
            attn_weight_for_v
                ->narrow({{0, 0, 1}, {1, 0, 1}, {2, 0, real_total_seq_len}})
                ->contiguous()
                ->to(infinicore::Device::cpu());
        auto capture_values =
            V->narrow(
                 {{0, 0, 1},
                  {1, 0, real_total_seq_len},
                  {2, 0, head_dim_}})
                ->contiguous()
                ->to(infinicore::Device::cpu());
        infinicore::context::syncStream();

        const std::string capture_prefix(v_contract_capture_dir);
        const std::string weights_path =
            capture_prefix + "/weights.bf16";
        const std::string values_path =
            capture_prefix + "/values.bf16";
        const std::string metadata_path =
            capture_prefix + "/metadata.json";

        std::ofstream weights_stream(
            weights_path,
            std::ios::out | std::ios::binary | std::ios::trunc);
        std::ofstream values_stream(
            values_path,
            std::ios::out | std::ios::binary | std::ios::trunc);
        std::ofstream metadata_stream(
            metadata_path,
            std::ios::out | std::ios::trunc);
        if (!weights_stream || !values_stream || !metadata_stream) {
            throw std::runtime_error(
                "failed to open V legacy-contract capture files");
        }
        weights_stream.write(
            reinterpret_cast<const char *>(capture_weights->data()),
            static_cast<std::streamsize>(
                capture_weights->numel() * sizeof(uint16_t)));
        values_stream.write(
            reinterpret_cast<const char *>(capture_values->data()),
            static_cast<std::streamsize>(
                capture_values->numel() * sizeof(uint16_t)));
        metadata_stream
            << "{\n"
            << "  \"dtype\": \"bf16\",\n"
            << "  \"batch_heads\": 1,\n"
            << "  \"kv_len\": " << real_total_seq_len << ",\n"
            << "  \"head_dim\": " << head_dim_ << ",\n"
            << "  \"layer\": " << layer_idx_ << ",\n"
            << "  \"q_len\": " << seq_len << "\n"
            << "}\n";
        if (!weights_stream || !values_stream || !metadata_stream) {
            throw std::runtime_error(
                "failed while writing V legacy-contract capture");
        }
        v_contract_capture_done = true;
        std::cerr
            << "INFINILM_V_LEGACY_CONTRACT_CAPTURE"
            << " layer=" << layer_idx_
            << " kv_len=" << real_total_seq_len
            << " head_dim=" << head_dim_
            << " dir=" << v_contract_capture_dir
            << std::endl;
    }

    auto out = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "attn_v_matmul", attn_weight_for_v);
        if (infiniLmVCacheDecodeLayoutEnabled() &&
            use_v_cache_decode_layout) {
            auto static_cache =
                std::dynamic_pointer_cast<cache::StaticKVCache>(kv_cache);
            if (static_cache != nullptr) {
                auto layout = static_cache->get_v_decode_layout(layer_idx_);
                auto layout_view = layout->view(
                    {batch_size * num_key_value_heads_,
                     layout->size(2),
                     head_dim_,
                     static_cache->v_decode_layout_page_size()});
                auto layout_out = infinicore::Tensor::empty(
                    {batch_size * num_attention_heads_, seq_len, head_dim_},
                    attn_weight_for_v->dtype(),
                    attn_weight_for_v->device());
                auto matvec_fn = infiniLmResolveVCacheDecodeLayoutMatvec();
                if (matvec_fn == nullptr) {
                    throw std::runtime_error(
                        "V-cache decode-layout matvec symbol is not available");
                }
                {
                    INFINILM_PROFILE_SCOPE(
                        phase,
                        static_cast<long>(layer_idx_),
                        rank_info_.tp_rank,
                        "attn_v_cache_decode_layout",
                        attn_weight_for_v);
                    const int status = matvec_fn(
                        layout_out->data(),
                        attn_weight_for_v->data(),
                        layout_view->data(),
                        real_total_seq_len,
                        batch_size * num_attention_heads_,
                        head_dim_,
                        layout_view->size(1),
                        static_cache->v_decode_layout_page_size(),
                        attn_weight_for_v->stride(0),
                        layout_view->stride(0),
                        layout_out->stride(0),
                        infiniLmVCacheDecodeLayoutVariant());
                    if (status != 0) {
                        throw std::runtime_error(
                            "V-cache decode-layout matvec failed");
                    }
                }
                if (layer_idx_ == 0) {
                    std::cerr
                        << "INFINILM_V_CACHE_DECODE_LAYOUT real_len="
                        << real_total_seq_len
                        << " total_len=" << total_seq_len
                        << " batch_heads="
                        << batch_size * num_attention_heads_
                        << " page_size="
                        << static_cache->v_decode_layout_page_size()
                        << " variant="
                        << infiniLmVCacheDecodeLayoutVariant()
                        << std::endl;
                }
                return layout_out;
            }
        }
        if (v_split_enabled) {
            const size_t align = infiniLmKvPadQkSplitAlign();
            const size_t prefix_len = (real_total_seq_len / align) * align;
            const size_t tail_len = real_total_seq_len - prefix_len;
            auto out_prefix = infinicore::op::matmul(
                attn_weight_for_v->narrow({{2, 0, prefix_len}}),
                V_for_matmul->narrow({{1, 0, prefix_len}}));
            if (tail_len == 0) {
                if (layer_idx_ == 0) {
                    std::cerr << "INFINILM_KV_PAD_V_SPLIT real_len="
                              << real_total_seq_len
                              << " prefix_len=" << prefix_len
                              << " tail_len=0 align=" << align
                              << std::endl;
                }
                return out_prefix;
            }
            auto attn_weight_tail = attn_weight_for_v->narrow({{2, prefix_len, tail_len}});
            auto V_tail = V_for_matmul->narrow({{1, prefix_len, tail_len}});
            if (infiniLmKvPadVSplitBetaAccumEnabled()) {
                // c = prefix + tail. This retains only the prefix BF16 store;
                // the tail is accumulated in the GEMM FP32 accumulator.
                infinicore::op::gemm_(out_prefix, attn_weight_tail, V_tail, 1.0f, 1.0f);
                if (layer_idx_ == 0) {
                    std::cerr << "INFINILM_KV_PAD_V_SPLIT_BETA_ACCUM real_len="
                              << real_total_seq_len
                              << " prefix_len=" << prefix_len
                              << " tail_len=" << tail_len
                              << " align=" << align
                              << std::endl;
                }
                return out_prefix;
            }
            auto out_tail = infinicore::op::matmul(attn_weight_tail, V_tail);
            if (layer_idx_ == 0) {
                std::cerr << "INFINILM_KV_PAD_V_SPLIT real_len="
                          << real_total_seq_len
                          << " prefix_len=" << prefix_len
                          << " tail_len=" << tail_len
                          << " align=" << align
                          << std::endl;
            }
            return infinicore::op::add(out_prefix, out_tail);
        }
        return infinicore::op::matmul(attn_weight_for_v, V_for_matmul); // [bs * n_kv_head, ng * seq_len, head_dim]
    }();
    if (seq_len == 1) {
        infiniLmAttentionLocalCertTrace(
            out,
            layer_idx_,
            real_total_seq_len,
            total_seq_len,
            batch_size * num_attention_heads_,
            head_dim_);
    }

    if (infiniLmKvPadNumericTraceEnabled() && seq_len == 1) {
        bool padded_mode = real_total_seq_len < total_seq_len;
        size_t trace_step = infiniLmLayerwiseTraceStep(padded_mode, true, layer_idx_);
        if (trace_step < infiniLmLayerwiseTraceLimit()) {
            uint64_t out_hash = infiniLmTraceTensorHash(out);
            std::cerr << "INFINILM_KV_PAD_LAYER_VOUT mode=" << (padded_mode ? "padded" : "baseline")
                      << " layer=" << layer_idx_
                      << " step=" << trace_step
                      << " numel=" << out->numel()
                      << " hash=" << out_hash
                      << std::endl;
        }
    }

    auto attn_output = iv_decode_layout::output(out, batch_size, num_attention_heads_, seq_len, head_dim_);

    auto output = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "attn_o_proj", attn_output);
        return o_proj_->forward(attn_output);
    }();

    return output;
}

infinicore::Tensor LlamaAttention::forward_paged_(const infinicore::Tensor &hidden_states,
                                                  const infinicore::Tensor &position_ids,
                                                  std::shared_ptr<infinilm::cache::PagedKVCache> paged_kv_cache,
                                                  std::optional<infinicore::Tensor> total_sequence_lengths,
                                                  std::optional<infinicore::Tensor> input_offsets,
                                                  std::optional<infinicore::Tensor> block_tables,
                                                  std::optional<infinicore::Tensor> slot_mapping) const {
    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());

    // Input shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // Only support batchsize==1, all requests should be flattened along seqlen dimension
    ASSERT_EQ(batch_size, 1);
    // Decode only if total_len == num_requests
    bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);

    const auto phase = infinilm::profile::phase_from_tensor(hidden_states);
    INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "attention_paged_total", hidden_states);

    // 1. Project Q, K, V
    auto [q, k, v] = [&]() {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "paged_attn_qkv_proj", hidden_states);
        return qkv_proj_->forward_split(hidden_states_mutable);
    }();

    // 2. Reshape for multi-head attention

    // Reshape Q, K, V to include batch dimension
    // Python: query_states = self.q_proj(hidden_states).view(querys_shape)
    // The view operation requires the tensor to be contiguous in the required dimensions
    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});

    if (use_qk_norm_) {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "paged_attn_qk_norm", q_reshaped);
        q_reshaped = q_norm_->forward(q_reshaped);
        k_reshaped = k_norm_->forward(k_reshaped);
    }

    // 3. Prepare position_ids for RoPE - align with Python pattern
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids;
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }

    // 4. Apply RoPE to Q and K
    {
        INFINILM_PROFILE_SCOPE(phase, static_cast<long>(layer_idx_), rank_info_.tp_rank, "paged_attn_rope", q_reshaped);
        rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true); // [bs, seq_len, n_q_head, head_dim]
        rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true); // [bs, seq_len, n_kv_head, head_dim]
    }

    //  5. Prepare KV caches
    //  Ensure contiguous after permute for F16 compatibility with cache operations
    auto [k_total, v_total] = paged_kv_cache->update(layer_idx_,
                                                     k_reshaped,
                                                     v_reshaped,
                                                     slot_mapping.value());

    // 6. Compute attention
    infinicore::Tensor attn_output = infinicore::Tensor::empty({seq_len, num_attention_heads_, head_dim_}, q_reshaped->dtype(), q_reshaped->device());

    if (is_prefill) {
        infinicore::op::paged_attention_prefill_(
            attn_output,
            q_reshaped,
            k_total,
            v_total,
            block_tables.value(),
            total_sequence_lengths.value(),
            input_offsets.value(),
            std::nullopt,
            scaling_);

    } else {
        infinicore::op::paged_attention_(
            attn_output,
            q_reshaped,
            k_total,
            v_total,
            block_tables.value(),
            total_sequence_lengths.value(),
            std::nullopt,
            scaling_);
    }

    // 7. Project output
    attn_output = attn_output->view({1, seq_len, num_attention_heads_ * head_dim_});
    return o_proj_->forward(attn_output);
}

infinicore::Tensor LlamaAttention::forward(const infinicore::Tensor &hidden_states,
                                           const infinicore::Tensor &position_ids,
                                           std::shared_ptr<cache::Cache> kv_cache,
                                           std::optional<infinicore::Tensor> past_sequence_lengths,
                                           std::optional<infinicore::Tensor> total_sequence_lengths,
                                           std::optional<infinicore::Tensor> input_offsets,
                                           std::optional<infinicore::Tensor> block_tables,
                                           std::optional<infinicore::Tensor> slot_mapping) const {
    if (!rotary_emb_) {
        throw std::runtime_error("LlamaAttention: rotary_emb not configured");
    }

    infinicore::Tensor output;
    if (auto paged_kv_cache = std::dynamic_pointer_cast<cache::PagedKVCache>(kv_cache)) {
        output = forward_paged_(hidden_states, position_ids, paged_kv_cache, total_sequence_lengths, input_offsets, block_tables, slot_mapping);
    } else {

        output = forward_(hidden_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths);
    }
    return output;
}

void LlamaAttention::set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
    rotary_emb_ = rotary_emb;
}

} // namespace infinilm::models::llama

// INFINILM_ATTENTION_STATE_FUSED_V2B_ATTR_TRACE_METADATA

// INFINILM_ATTENTION_STATE_FUSED_V2B_LAYER_FALLBACK
