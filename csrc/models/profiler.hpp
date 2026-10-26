#pragma once

#include "infinicore/context/context.hpp"
#include "infinicore/tensor.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace infinilm::profile {

inline bool env_flag(const char *name, bool default_value = false) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }
    std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

inline int env_int(const char *name, int default_value = -1) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }
    try {
        return std::stoi(value);
    } catch (...) {
        return default_value;
    }
}

inline bool enabled() {
    static bool value = env_flag("INFINILM_PROFILE", false);
    return value;
}

inline bool sync_enabled() {
    static bool value = env_flag("INFINILM_PROFILE_SYNC", true);
    return value;
}

inline bool should_profile_layer(long layer_idx) {
    static int layer_filter = env_int("INFINILM_PROFILE_LAYER", -1);
    return layer_filter < 0 || layer_idx < 0 || layer_filter == layer_idx;
}

inline std::string shape_to_string(const infinicore::Tensor &tensor) {
    std::ostringstream oss;
    oss << "[";
    if (tensor) {
        const auto shape = tensor->shape();
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i != 0) {
                oss << "x";
            }
            oss << shape[i];
        }
    }
    oss << "]";
    return oss.str();
}

inline std::string phase_from_tensor(const infinicore::Tensor &tensor) {
    if (!tensor) {
        return "unknown";
    }
    const auto shape = tensor->shape();
    if (shape.size() >= 2 && shape[1] > 1) {
        return "prefill";
    }
    return "decode";
}

inline std::mutex &log_mutex() {
    static std::mutex mutex;
    return mutex;
}

class ScopedTimer {
public:
    ScopedTimer(std::string phase,
                long layer_idx,
                int rank,
                std::string op,
                const infinicore::Tensor &tensor)
        : phase_(std::move(phase)),
          layer_idx_(layer_idx),
          rank_(rank),
          op_(std::move(op)),
          shape_(shape_to_string(tensor)),
          active_(enabled() && should_profile_layer(layer_idx)) {
        if (!active_) {
            return;
        }
        if (sync_enabled()) {
            infinicore::context::syncDevice();
        }
        start_ = std::chrono::steady_clock::now();
    }

    ~ScopedTimer() {
        if (!active_) {
            return;
        }
        if (sync_enabled()) {
            infinicore::context::syncDevice();
        }
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        std::lock_guard<std::mutex> guard(log_mutex());
        std::cout << "INF_PROFILE"
                  << " phase=" << phase_
                  << " layer=" << layer_idx_
                  << " rank=" << rank_
                  << " op=" << op_
                  << " ms=" << ms
                  << " shape=" << shape_
                  << std::endl;
    }

private:
    std::string phase_;
    long layer_idx_;
    int rank_;
    std::string op_;
    std::string shape_;
    bool active_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace infinilm::profile

#define INFINILM_PROFILE_CONCAT_INNER(a, b) a##b
#define INFINILM_PROFILE_CONCAT(a, b) INFINILM_PROFILE_CONCAT_INNER(a, b)
#define INFINILM_PROFILE_SCOPE(phase, layer, rank, op, tensor) \
    infinilm::profile::ScopedTimer INFINILM_PROFILE_CONCAT(_infinilm_profile_timer_, __LINE__)(phase, layer, rank, op, tensor)
