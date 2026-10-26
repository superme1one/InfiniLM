#pragma once
#include <cstddef>

namespace iv_rope_policy {
// Only the BF16 shapes whose original kernel was validated on TG-V200.
constexpr bool eligible(int original, size_t table, size_t batch, size_t seq,
                        size_t heads, bool bf16) {
    return original == 2048 && table == 64 && batch >= 1 && batch <= 16 &&
           seq >= 1 && seq <= 2048 && heads == 28 && bf16;
}
constexpr int threads(int original, size_t table, size_t batch, size_t seq,
                      size_t heads, bool bf16, bool enabled) {
    return enabled && eligible(original, table, batch, seq, heads, bf16) ? 64 : original;
}
}
