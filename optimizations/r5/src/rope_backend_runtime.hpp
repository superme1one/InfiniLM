#pragma once
#include "rope_backend_policy.hpp"
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace iv_rope_backend {
static std::atomic<unsigned long long> fast{0}, fallback{0}, prefill{0}, decode{0}, rejected{0};
static bool enabled() {
    static const bool v=[] {const char *p=std::getenv("IV_OPT_ROPE_BACKEND");return p && std::strcmp(p,"1")==0;}();
    return v;
}
static int select(int original, size_t table, size_t batch, size_t seq, size_t heads, bool bf16) {
    const bool valid=iv_rope_policy::eligible(original,table,batch,seq,heads,bf16);
    const int chosen=iv_rope_policy::threads(original,table,batch,seq,heads,bf16,enabled());
    if (enabled() && valid) ++fast; else ++fallback;
    if (!valid) ++rejected;
    if (seq==1) ++decode; else ++prefill;
    return chosen;
}
}
#define IV_ROPE_EXPORT extern "C" __attribute__((visibility("default"))) unsigned long long
IV_ROPE_EXPORT iv_rope_backend_fast(){return iv_rope_backend::fast.load();}
IV_ROPE_EXPORT iv_rope_backend_fallback(){return iv_rope_backend::fallback.load();}
IV_ROPE_EXPORT iv_rope_backend_prefill(){return iv_rope_backend::prefill.load();}
IV_ROPE_EXPORT iv_rope_backend_decode(){return iv_rope_backend::decode.load();}
IV_ROPE_EXPORT iv_rope_backend_rejected(){return iv_rope_backend::rejected.load();}
#undef IV_ROPE_EXPORT
