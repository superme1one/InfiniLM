#pragma once
#include <cstddef>

namespace iv_decode_layout {
// Deliberately narrower than a general contiguous predicate. No overflow: the
// batch/head/dimension bounds are checked before products are evaluated.
template<class Shape, class Stride>
bool eligible(const Shape &shape, const Stride &stride, size_t b, size_t h,
              size_t s, size_t d, bool bf16, bool iluvatar) {
    return b >= 1 && b <= 16 && h == 28 && s == 1 && d == 128 && bf16 && iluvatar &&
           shape.size() == 3 && stride.size() == 3 &&
           static_cast<size_t>(shape[0]) == b*h && shape[1] == 1 && static_cast<size_t>(shape[2]) == d &&
           stride[0] == 128 && stride[1] == 128 && stride[2] == 1;
}
}
