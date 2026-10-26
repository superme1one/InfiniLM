#pragma once
#include "decode_layout_guard.hpp"
#include <infinicore/tensor.hpp>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace iv_decode_layout {
static std::atomic<unsigned long long> fast{0}, fallback{0}, saved_bytes{0};
static bool enabled() {
    static const bool value=[] {const char *v=std::getenv("IV_OPT_DECODE_LAYOUT");return v && std::strcmp(v,"1")==0;}();
    return value;
}
static infinicore::Tensor output(infinicore::Tensor out, size_t b, size_t h, size_t s, size_t d) {
    if (enabled() && eligible(out->shape(),out->strides(),b,h,s,d,
            out->dtype()==infinicore::DataType::BF16,
            out->device().getType()==infinicore::Device::Type::ILUVATAR)) {
        // With s=1, flattening [b*h,1,d] has exactly the same element order as
        // view([b,h,1,d])->permute([0,2,1,3])->contiguous()->view([b,1,h*d]).
        auto view=out->view({b,1,h*d});
        if (view->data()!=out->data() || !view->is_contiguous())
            throw std::runtime_error("Decode layout view contract violated");
        ++fast;saved_bytes.fetch_add(b*h*d*2,std::memory_order_relaxed);
        return view;
    }
    ++fallback;
    return out->view({b,h,s,d})->permute({0,2,1,3})->contiguous()->view({b,s,h*d});
}
}
extern "C" unsigned long long iv_decode_layout_fast_calls(){return iv_decode_layout::fast.load();}
extern "C" unsigned long long iv_decode_layout_fallback_calls(){return iv_decode_layout::fallback.load();}
extern "C" unsigned long long iv_decode_layout_saved_bytes(){return iv_decode_layout::saved_bytes.load();}
