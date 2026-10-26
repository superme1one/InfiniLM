#pragma once
#include <cstdlib>
#include <cstring>
// Baseline keeps the two original nn::RoPE calls. No new cache or readback.
static bool iv_rope_pair_model_enabled() {
    static const bool enabled = [] {
        const char *v = std::getenv("IV_OPT_ROPE_PAIR");
        return v && std::strcmp(v, "1") == 0;
    }();
    return enabled;
}
