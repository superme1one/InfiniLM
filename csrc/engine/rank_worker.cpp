#include "last_logits_scope.hpp"
#include "rank_worker.hpp"

#include "../models/model_factory.hpp"

#include "infinicore/ops.hpp"

#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <string>
#include <map>
#include <cmath>
#include <vector>
#include <spdlog/spdlog.h>
#include <stdexcept>


namespace {
bool infiniLmKvPadNumericTraceEnabled() {
    const char *env = std::getenv("INFINILM_KV_PAD_NUMERIC_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}


bool infiniLmQkLogitsDiffTraceEnabled() {
    const char *env = std::getenv("INFINILM_QK_LOGITS_DIFF_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

size_t infiniLmQkLogitsDiffStepLimit() {
    const char *env = std::getenv("INFINILM_QK_LOGITS_DIFF_STEPS");
    if (env == nullptr || env[0] == '\0') {
        return 128;
    }
    return static_cast<size_t>(std::strtoull(env, nullptr, 10));
}

std::vector<size_t> infiniLmQkTopKIds(
    const std::vector<float> &values,
    size_t k) {
    k = std::min(k, values.size());
    std::vector<size_t> ids(k, 0);
    std::vector<float> scores(k, -std::numeric_limits<float>::infinity());
    for (size_t index = 0; index < values.size(); ++index) {
        float value = values[index];
        if (k == 0 || value <= scores[k - 1]) {
            continue;
        }
        size_t position = k - 1;
        while (position > 0 && value > scores[position - 1]) {
            scores[position] = scores[position - 1];
            ids[position] = ids[position - 1];
            --position;
        }
        scores[position] = value;
        ids[position] = index;
    }
    return ids;
}

size_t infiniLmQkTopKOverlap(
    const std::vector<size_t> &lhs,
    const std::vector<size_t> &rhs) {
    size_t overlap = 0;
    for (size_t left : lhs) {
        if (std::find(rhs.begin(), rhs.end(), left) != rhs.end()) {
            ++overlap;
        }
    }
    return overlap;
}


bool infiniLmTeacherForcedLogitsTraceEnabled() {
    const char *env = std::getenv("INFINILM_TEACHER_FORCED_LOGITS_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

std::string infiniLmTeacherForcedTraceEnv(
    const char *name,
    const char *fallback) {
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return std::string(fallback);
    }
    return std::string(env);
}

size_t infiniLmTeacherForcedTraceStep() {
    const char *env = std::getenv("INFINILM_TEACHER_FORCED_TRACE_STEP");
    if (env == nullptr || env[0] == '\0') {
        return 0;
    }
    return static_cast<size_t>(std::strtoull(env, nullptr, 10));
}

std::vector<size_t> infiniLmTeacherTopKIds(
    const std::vector<float> &values,
    size_t k) {
    k = std::min(k, values.size());
    std::vector<size_t> ids(k, 0);
    std::vector<float> scores(k, -std::numeric_limits<float>::infinity());
    for (size_t index = 0; index < values.size(); ++index) {
        float value = values[index];
        if (k == 0 || value <= scores[k - 1]) {
            continue;
        }
        size_t position = k - 1;
        while (position > 0 && value > scores[position - 1]) {
            scores[position] = scores[position - 1];
            ids[position] = ids[position - 1];
            --position;
        }
        scores[position] = value;
        ids[position] = index;
    }
    return ids;
}

size_t infiniLmTeacherTopKOverlap(
    const std::vector<size_t> &lhs,
    const std::vector<size_t> &rhs) {
    size_t overlap = 0;
    for (size_t left : lhs) {
        if (std::find(rhs.begin(), rhs.end(), left) != rhs.end()) {
            ++overlap;
        }
    }
    return overlap;
}

double infiniLmTeacherMargin(
    const std::vector<float> &values,
    const std::vector<size_t> &top_ids) {
    if (top_ids.size() < 2) {
        return 0.0;
    }
    return static_cast<double>(values[top_ids[0]]) -
           static_cast<double>(values[top_ids[1]]);
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
} // namespace

namespace infinilm::engine {

RankWorker::RankWorker(const InfinilmModel::Config &model_config,
                       const distributed::RankInfo &rank_info,
                       const cache::CacheConfig *cache_config)
    : model_config_(model_config),
      rank_info_(rank_info),
      job_cmd_(Command::INIT),
      has_job_(false),
      job_done_(false),
      should_exit_(false),
      init_done_(false),
      rng_(std::random_device{}()) {
    if (cache_config != nullptr) {
        pending_cache_config_ = cache_config->unique_copy();
    }
    // start the thread
    thread_ = std::thread(&RankWorker::thread_loop, this);

    // Wait until the worker thread finishes initialization (model created)
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return init_done_; });
}

std::string RankWorker::info() const {
    std::stringstream ss;

    ss << "RankWorker{";

    // Rank related
    ss << rank_info_.to_string() << " ";

    // Flags
    ss << "| init_done: " << (init_done_ ? "true" : "false") << " ";
    ss << "| should_exit: " << (should_exit_ ? "true" : "false") << " ";
    ss << "| has_job: " << (has_job_ ? "true" : "false") << " ";
    ss << "| job_done: " << (job_done_ ? "true" : "false") << " ";

    ss << "}";

    return ss.str();
}

//------------------------------------------------------
// load_param -- synchronous (blocks until worker finishes loading)
//------------------------------------------------------
void RankWorker::load_param(const std::string &name,
                            const infinicore::Tensor &param) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // If the worker is stopping, don't submit new jobs.
        if (should_exit_) {
            throw std::runtime_error("RankWorker is closing; cannot load_param");
        }

        pending_param_name_ = name;
        pending_param_ = param;

        job_cmd_ = Command::LOAD;
        has_job_ = true;
        job_done_ = false;
    }
    cv_.notify_all();

    // Wait for job completion
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return job_done_ || should_exit_; });

    if (should_exit_) {
        throw std::runtime_error("RankWorker stopped while loading parameter");
    }
}

//------------------------------------------------------
// state_dict --
//------------------------------------------------------
std::unordered_map<std::string, infinicore::nn::Parameter> RankWorker::state_dict() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return init_done_ || should_exit_; });

    if (!model_) {
        throw std::runtime_error("state_dict called before model initialization");
    }

    return model_->state_dict();
}

//------------------------------------------------------
// run -- asynchronous
//------------------------------------------------------
void RankWorker::run(const Input &args) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (should_exit_) {
        throw std::runtime_error("RankWorker is closing; cannot run");
    }

    pending_args_ = args;
    job_cmd_ = Command::RUN;
    has_job_ = true;
    job_done_ = false;

    cv_.notify_all();
}

//------------------------------------------------------
// wait -- asynchronous
//------------------------------------------------------
void RankWorker::wait() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return job_done_ || should_exit_; });

    if (should_exit_) {
        throw std::runtime_error("RankWorker stopped during run");
    }
}

void RankWorker::reset_cache(const cache::CacheConfig *new_config) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (should_exit_) {
        throw std::runtime_error("RankWorker is closing; cannot reset_cache");
    }

    // Store both the position and the new config
    pending_cache_config_ = new_config->unique_copy();
    job_cmd_ = Command::RESET_CACHE;
    has_job_ = true;
    job_done_ = false;
    cv_.notify_all();
}

//------------------------------------------------------
// close -- request shutdown and join thread
//------------------------------------------------------
void RankWorker::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        should_exit_ = true;
        has_job_ = false; // don't keep old jobs pending
        job_cmd_ = Command::STOP;
    }
    cv_.notify_all();

    if (thread_.joinable()) {
        thread_.join();
    }
}

//------------------------------------------------------
// get_output (thread safe)
//------------------------------------------------------
RankWorker::Output RankWorker::get_output() {
    std::lock_guard<std::mutex> lock(mutex_);
    return output_;
}

//------------------------------------------------------
// thread_loop
//------------------------------------------------------
void RankWorker::thread_loop() {
    try {
        {
            std::lock_guard<std::mutex> lk(mutex_);

            // Initialize device & model outside of holding the main mutex to avoid blocking callers.
            infinicore::context::setDevice(rank_info_.device);

            // Create model using factory (may be expensive)
            model_ = InfinilmModelFactory::createModel(model_config_, rank_info_, pending_cache_config_ != nullptr ? pending_cache_config_.get() : nullptr);
            if (!model_) {
                throw std::runtime_error("Failed to create model");
            }
            init_done_ = true;
        }
        cv_.notify_all();

        // Main loop: wait for jobs or exit
        while (true) {
            Command local_cmd = Command::INIT;
            std::string local_param_name;
            infinicore::Tensor local_param;
            Input local_args;
            std::unique_ptr<cache::CacheConfig> local_cache_config;

            // Wait for a job or exit
            {
                std::unique_lock<std::mutex> lk(mutex_);
                cv_.wait(lk, [&] { return has_job_ || should_exit_; });

                if (should_exit_) {
                    break;
                }

                // capture job data and clear has_job_
                local_cmd = job_cmd_;
                if (local_cmd == Command::LOAD) {
                    local_param_name = pending_param_name_;
                    local_param = pending_param_;
                } else if (local_cmd == Command::RUN) {
                    local_args = pending_args_;
                } else if (local_cmd == Command::RESET_CACHE) {
                    if (pending_cache_config_ != nullptr) {
                        local_cache_config = pending_cache_config_->unique_copy();
                    }
                }
                // mark job as being processed
                has_job_ = false;
                job_done_ = false;
            } // unlock mutex while executing the job

            // Execute job outside the lock
            if (local_cmd == Command::LOAD) {
                try {
                    model_->load_parameter(local_param_name, local_param);
                } catch (const std::exception &e) {
                    // convert exceptions to a safe behavior: set should_exit_ and notify caller
                    std::lock_guard<std::mutex> lk(mutex_);
                    should_exit_ = true;
                    job_done_ = true;
                    cv_.notify_all();
                    // rethrow so the thread can be joined and caller sees an error if desired (optional)
                    spdlog::error("[{}] exception during load_parameter_: {}\n", info(), e.what());
                    break;
                }

                // signal completion
                {
                    std::lock_guard<std::mutex> lk(mutex_);
                    job_done_ = true;
                }
                cv_.notify_all();

            } else if (local_cmd == Command::RUN) {
                try {
                    {
                        std::lock_guard<std::mutex> lk(mutex_);

                        iv_last_logits::Scope logits_scope(local_args.input_ids, local_args.input_offsets,
                            rank_info_.tp_size == 1 && !local_args.block_tables.has_value() && !local_args.slot_mapping.has_value());
                        auto model_args = local_args.to_model_input(rank_info_.device);
                        // Forward calculation
                        auto logits{model_->forward(model_args).logits};
                        logits_scope.observe(logits);
                        // Random sampling (rank 0 only)
                        if (rank_info_.tp_rank == 0) {
                            auto temperature{local_args.temperature};
                            auto top_p{local_args.top_p};
                            auto top_k{local_args.top_k};

                            const auto &logits_shape{logits->shape()};
                            const auto &vocab_size{logits_shape[2]};
                            const auto &total_len{logits_shape[1]};
                            const auto &batch_size{logits_shape[0]};

                            auto n_req = local_args.input_offsets.value()->size(0) - 1;
                            int64_t *input_offsets = (int64_t *)local_args.input_offsets.value()->data();

                            auto output_ids{infinicore::Tensor::empty({n_req}, infinicore::DataType::I64, rank_info_.device)};

                            for (auto i{decltype(n_req)(0)}; i < n_req; ++i) {
                                auto score{logits->view({batch_size * total_len, vocab_size})->narrow({{0, (logits_scope.compact() ? size_t(i) : size_t(input_offsets[i + 1] - 1)), 1}})->view({vocab_size})};
                                if (i == 0 && infiniLmKvPadNumericTraceEnabled()) {
                                    static thread_local size_t baseline_logits_step = 0;
                                    static thread_local size_t padded_logits_step = 0;
                                    const char *align_env = std::getenv("INFINILM_KV_PAD_ALIGN");
                                    const char *shared_prefix_env =
                                        std::getenv("INFINILM_SHARED_PREFIX_KV_REUSE");
                                    bool shared_prefix_candidate_mode =
                                        shared_prefix_env != nullptr &&
                                        shared_prefix_env[0] != '\0' &&
                                        shared_prefix_env[0] != '0';
                                    bool padded_mode = align_env != nullptr && align_env[0] != '\0' && align_env[0] != '0';
                                    size_t trace_step = padded_mode ? padded_logits_step++ : baseline_logits_step++;
                                    auto score_cpu = score->contiguous()->to(infinicore::Device::cpu());
                                    infinicore::context::syncStream();
                                    const std::byte *raw = score_cpu->data();
                                    auto dtype = score_cpu->dtype();
                                    size_t top1_id = 0;
                                    size_t top2_id = 0;
                                    float top1 = -std::numeric_limits<float>::infinity();
                                    float top2 = -std::numeric_limits<float>::infinity();
                                    for (size_t token_id = 0; token_id < vocab_size; ++token_id) {
                                        float value = infiniLmTraceValue(raw, dtype, token_id);
                                        if (value > top1) {
                                            top2 = top1;
                                            top2_id = top1_id;
                                            top1 = value;
                                            top1_id = token_id;
                                        } else if (value > top2) {
                                            top2 = value;
                                            top2_id = token_id;
                                        }
                                    }
                                    std::cerr << "INFINILM_KV_PAD_LOGITS_NUMERIC mode=" << (padded_mode ? "padded" : "baseline")
                                              << " step=" << trace_step
                                              << " top1_id=" << top1_id
                                              << " top1=" << top1
                                              << " top2_id=" << top2_id
                                              << " top2=" << top2
                                              << " margin=" << (top1 - top2)
                                              << std::endl;
                                }

                                if (i == 0 && infiniLmQkLogitsDiffTraceEnabled()) {
                                    static thread_local size_t baseline_diff_step = 0;
                                    static thread_local size_t candidate_diff_step = 0;
                                    static thread_local std::vector<std::vector<float>> baseline_logits;
                                    const char *align_env = std::getenv("INFINILM_KV_PAD_ALIGN");
                                    const char *mode_env = std::getenv("INFINILM_KV_PAD_MODE");
                                    bool attention_state_fused_mode =
                                        mode_env != nullptr &&
                                        std::strcmp(mode_env, "attention_state_fused") == 0;
                                    bool attention_state_fused_v2_mode =
                                        mode_env != nullptr &&
                                        std::strcmp(mode_env, "attention_state_fused_v2") == 0;
                                    bool attention_state_fused_v2b_mode =
                                        mode_env != nullptr &&
                                        std::strcmp(mode_env, "attention_state_fused_v2b") == 0;
                                    bool candidate_mode = attention_state_fused_mode ||
                                                          attention_state_fused_v2_mode ||
                                                          attention_state_fused_v2b_mode ||
                                                          (align_env != nullptr &&
                                                           align_env[0] != '\0' &&
                                                           align_env[0] != '0');
                                    const char *shared_prefix_diff_env =
                                        std::getenv("INFINILM_SHARED_PREFIX_KV_REUSE");
                                    candidate_mode = candidate_mode ||
                                        (shared_prefix_diff_env != nullptr &&
                                         shared_prefix_diff_env[0] != '\0' &&
                                         shared_prefix_diff_env[0] != '0');
                                    size_t trace_step = candidate_mode
                                        ? candidate_diff_step++
                                        : baseline_diff_step++;
                                    if (trace_step < infiniLmQkLogitsDiffStepLimit()) {
                                        auto score_cpu = score->contiguous()->to(infinicore::Device::cpu());
                                        infinicore::context::syncStream();
                                        const std::byte *raw = score_cpu->data();
                                        auto dtype = score_cpu->dtype();
                                        std::vector<float> values(vocab_size, 0.0f);
                                        for (size_t token_id = 0; token_id < vocab_size; ++token_id) {
                                            values[token_id] = infiniLmTraceValue(raw, dtype, token_id);
                                        }

                                        if (!candidate_mode) {
                                            if (baseline_logits.size() <= trace_step) {
                                                baseline_logits.resize(trace_step + 1);
                                            }
                                            baseline_logits[trace_step] = std::move(values);
                                            std::cerr << "INFINILM_QK_LOGITS_BASELINE step=" << trace_step
                                                      << " count=" << vocab_size
                                                      << std::endl;
                                        } else if (trace_step >= baseline_logits.size() ||
                                                   baseline_logits[trace_step].size() != values.size()) {
                                            std::cerr << "INFINILM_QK_LOGITS_DIFF_MISSING step=" << trace_step
                                                      << " count=" << values.size()
                                                      << std::endl;
                                        } else {
                                            const auto &reference = baseline_logits[trace_step];
                                            double sum_abs = 0.0;
                                            double sum_sq = 0.0;
                                            double max_abs = 0.0;
                                            double dot = 0.0;
                                            double norm_ref = 0.0;
                                            double norm_candidate = 0.0;
                                            size_t unequal = 0;
                                            for (size_t token_id = 0; token_id < values.size(); ++token_id) {
                                                double lhs = static_cast<double>(reference[token_id]);
                                                double rhs = static_cast<double>(values[token_id]);
                                                double diff = std::abs(lhs - rhs);
                                                unequal += diff != 0.0 ? 1 : 0;
                                                sum_abs += diff;
                                                sum_sq += diff * diff;
                                                max_abs = std::max(max_abs, diff);
                                                dot += lhs * rhs;
                                                norm_ref += lhs * lhs;
                                                norm_candidate += rhs * rhs;
                                            }
                                            auto base_top10 = infiniLmQkTopKIds(reference, 10);
                                            auto candidate_top10 = infiniLmQkTopKIds(values, 10);
                                            std::vector<size_t> base_top5(
                                                base_top10.begin(),
                                                base_top10.begin() + std::min<size_t>(5, base_top10.size()));
                                            std::vector<size_t> candidate_top5(
                                                candidate_top10.begin(),
                                                candidate_top10.begin() + std::min<size_t>(5, candidate_top10.size()));
                                            double cosine = 0.0;
                                            if (norm_ref > 0.0 && norm_candidate > 0.0) {
                                                cosine = dot / std::sqrt(norm_ref * norm_candidate);
                                            }
                                            double baseline_margin = 0.0;
                                            if (base_top10.size() >= 2) {
                                                baseline_margin =
                                                    static_cast<double>(reference[base_top10[0]]) -
                                                    static_cast<double>(reference[base_top10[1]]);
                                            }
                                            std::cerr << "INFINILM_QK_LOGITS_DIFF step=" << trace_step
                                                      << " count=" << values.size()
                                                      << " unequal=" << unequal
                                                      << " unequal_ratio="
                                                      << (static_cast<double>(unequal) / values.size())
                                                      << " mean_abs=" << (sum_abs / values.size())
                                                      << " max_abs=" << max_abs
                                                      << " rmse=" << std::sqrt(sum_sq / values.size())
                                                      << " cosine=" << cosine
                                                      << " baseline_top1="
                                                      << (base_top10.empty() ? 0 : base_top10[0])
                                                      << " candidate_top1="
                                                      << (candidate_top10.empty() ? 0 : candidate_top10[0])
                                                      << " baseline_margin=" << baseline_margin
                                                      << " top5_overlap="
                                                      << infiniLmQkTopKOverlap(base_top5, candidate_top5)
                                                      << " top10_overlap="
                                                      << infiniLmQkTopKOverlap(base_top10, candidate_top10)
                                                      << std::endl;
                                        }
                                    }
                                }

                                if (infiniLmTeacherForcedLogitsTraceEnabled()) {
                                    using TeacherTraceKey = std::pair<size_t, size_t>;
                                    static thread_local std::map<TeacherTraceKey, std::vector<float>>
                                        teacher_forced_baseline_logits;

                                    const std::string teacher_role = infiniLmTeacherForcedTraceEnv(
                                        "INFINILM_TEACHER_FORCED_TRACE_ROLE",
                                        "candidate");
                                    const std::string teacher_tag = infiniLmTeacherForcedTraceEnv(
                                        "INFINILM_TEACHER_FORCED_TRACE_TAG",
                                        "unknown");
                                    const size_t teacher_step = infiniLmTeacherForcedTraceStep();
                                    const size_t teacher_batch = static_cast<size_t>(i);

                                    auto score_cpu = score->contiguous()->to(infinicore::Device::cpu());
                                    infinicore::context::syncStream();
                                    const std::byte *raw = score_cpu->data();
                                    auto dtype = score_cpu->dtype();
                                    std::vector<float> values(vocab_size, 0.0f);
                                    for (size_t token_id = 0; token_id < vocab_size; ++token_id) {
                                        values[token_id] = infiniLmTraceValue(raw, dtype, token_id);
                                    }

                                    const TeacherTraceKey teacher_key{teacher_step, teacher_batch};
                                    if (teacher_role == "baseline") {
                                        teacher_forced_baseline_logits[teacher_key] = std::move(values);
                                        auto base_top10 = infiniLmTeacherTopKIds(
                                            teacher_forced_baseline_logits[teacher_key],
                                            10);
                                        auto base_top2 = infiniLmTeacherTopKIds(
                                            teacher_forced_baseline_logits[teacher_key],
                                            2);
                                        std::cerr << "INFINILM_TEACHER_FORCED_LOGITS_BASELINE"
                                                  << " tag=" << teacher_tag
                                                  << " step=" << teacher_step
                                                  << " batch=" << teacher_batch
                                                  << " count=" << vocab_size
                                                  << " baseline_top1="
                                                  << (base_top10.empty() ? 0 : base_top10[0])
                                                  << " baseline_top2="
                                                  << (base_top2.size() < 2 ? 0 : base_top2[1])
                                                  << " baseline_margin="
                                                  << infiniLmTeacherMargin(
                                                         teacher_forced_baseline_logits[teacher_key],
                                                         base_top2)
                                                  << std::endl;
                                    } else {
                                        auto baseline_it = teacher_forced_baseline_logits.find(teacher_key);
                                        if (baseline_it == teacher_forced_baseline_logits.end() ||
                                            baseline_it->second.size() != values.size()) {
                                            std::cerr << "INFINILM_TEACHER_FORCED_LOGITS_MISSING"
                                                      << " tag=" << teacher_tag
                                                      << " step=" << teacher_step
                                                      << " batch=" << teacher_batch
                                                      << " count=" << values.size()
                                                      << std::endl;
                                        } else {
                                            const auto &reference = baseline_it->second;
                                            double sum_abs = 0.0;
                                            double sum_sq = 0.0;
                                            double max_abs = 0.0;
                                            double dot = 0.0;
                                            double norm_ref = 0.0;
                                            double norm_candidate = 0.0;
                                            size_t unequal = 0;
                                            for (size_t token_id = 0; token_id < values.size(); ++token_id) {
                                                const double lhs = static_cast<double>(reference[token_id]);
                                                const double rhs = static_cast<double>(values[token_id]);
                                                const double diff = std::abs(lhs - rhs);
                                                unequal += diff != 0.0 ? 1 : 0;
                                                sum_abs += diff;
                                                sum_sq += diff * diff;
                                                max_abs = std::max(max_abs, diff);
                                                dot += lhs * rhs;
                                                norm_ref += lhs * lhs;
                                                norm_candidate += rhs * rhs;
                                            }

                                            auto base_top10 = infiniLmTeacherTopKIds(reference, 10);
                                            auto candidate_top10 = infiniLmTeacherTopKIds(values, 10);
                                            auto base_top2 = infiniLmTeacherTopKIds(reference, 2);
                                            auto candidate_top2 = infiniLmTeacherTopKIds(values, 2);
                                            std::vector<size_t> base_top5(
                                                base_top10.begin(),
                                                base_top10.begin() + std::min<size_t>(5, base_top10.size()));
                                            std::vector<size_t> candidate_top5(
                                                candidate_top10.begin(),
                                                candidate_top10.begin() + std::min<size_t>(5, candidate_top10.size()));

                                            double cosine = 0.0;
                                            if (norm_ref > 0.0 && norm_candidate > 0.0) {
                                                cosine = dot / std::sqrt(norm_ref * norm_candidate);
                                            }

                                            const size_t baseline_top1 = base_top10.empty() ? 0 : base_top10[0];
                                            const size_t candidate_top1 = candidate_top10.empty() ? 0 : candidate_top10[0];
                                            const double baseline_top1_delta =
                                                static_cast<double>(values[baseline_top1]) -
                                                static_cast<double>(reference[baseline_top1]);

                                            std::cerr << "INFINILM_TEACHER_FORCED_LOGITS_DIFF"
                                                      << " tag=" << teacher_tag
                                                      << " step=" << teacher_step
                                                      << " batch=" << teacher_batch
                                                      << " count=" << values.size()
                                                      << " unequal=" << unequal
                                                      << " unequal_ratio="
                                                      << (static_cast<double>(unequal) / values.size())
                                                      << " mean_abs=" << (sum_abs / values.size())
                                                      << " max_abs=" << max_abs
                                                      << " rmse=" << std::sqrt(sum_sq / values.size())
                                                      << " cosine=" << cosine
                                                      << " baseline_top1=" << baseline_top1
                                                      << " baseline_top2="
                                                      << (base_top2.size() < 2 ? 0 : base_top2[1])
                                                      << " baseline_margin="
                                                      << infiniLmTeacherMargin(reference, base_top2)
                                                      << " candidate_top1=" << candidate_top1
                                                      << " candidate_top2="
                                                      << (candidate_top2.size() < 2 ? 0 : candidate_top2[1])
                                                      << " candidate_margin="
                                                      << infiniLmTeacherMargin(values, candidate_top2)
                                                      << " baseline_candidate_top1_delta="
                                                      << baseline_top1_delta
                                                      << " top5_overlap="
                                                      << infiniLmTeacherTopKOverlap(base_top5, candidate_top5)
                                                      << " top10_overlap="
                                                      << infiniLmTeacherTopKOverlap(base_top10, candidate_top10)
                                                      << std::endl;
                                        }
                                    }
                                }
                                auto out{output_ids->narrow({{0, i, 1}})->view({})};
                                float random_val = std::uniform_real_distribution<float>(0, 1)(rng_);
                                infinicore::op::random_sample_(
                                    out, score, random_val, top_p, top_k, temperature);
                            }

                            output_ids = output_ids->to(infinicore::Device::cpu());

                            infinicore::context::syncStream();

                            auto out{Output{output_ids}};

                            output_ = std::move(out);
                        }

                        job_done_ = true;
                    }
                    cv_.notify_all();

                } catch (const std::exception &e) {
                    std::lock_guard<std::mutex> lk(mutex_);
                    should_exit_ = true;
                    job_done_ = true;
                    cv_.notify_all();
                    spdlog::error("[{}] exception during forward: {}\n", info(), e.what());
                    break;
                }
            } else if (local_cmd == Command::RESET_CACHE) {
                try {
                    model_->reset_cache(local_cache_config != nullptr ? local_cache_config.get() : nullptr);

                    {
                        std::lock_guard<std::mutex> lk(mutex_);
                        job_done_ = true;
                    }
                    cv_.notify_all();

                } catch (const std::exception &e) {
                    std::lock_guard<std::mutex> lk(mutex_);
                    should_exit_ = true;
                    job_done_ = true;
                    cv_.notify_all();
                    spdlog::error("[{}] exception during reset_cache: {}\n", info(), e.what());
                    break;
                }
            } else {
                // Shouldn't reach here (no-op)
            }
        } // while
    } catch (const std::exception &e) {
        // Top-level exception: ensure any waiters are woken and the thread exits cleanly.
        {
            std::lock_guard<std::mutex> lk(mutex_);
            should_exit_ = true;
            job_done_ = true;
        }
        cv_.notify_all();
        spdlog::error("[{}] fatal exception in thread_loop: {} \n", info(), e.what());
    }
}

} // namespace infinilm::engine
