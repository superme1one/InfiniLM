#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace iv_rope_pair_guard {
using Strides=std::array<int64_t,4>;
inline bool shape(size_t b,size_t s,size_t h,size_t d) {
    return b>=1 && b<=16 && s>=1 && s<=2048 && h==28 && d==128;
}
// The QKV split is either dense or a slice of a row containing Q,K,V.
inline bool input(size_t s,const Strides &t) {
    return t[3]==1 && t[2]==128 && (t[1]==3584 || t[1]==10752) && t[0]==int64_t(s)*t[1];
}
inline bool output(size_t s,const Strides &t) {
    return t==Strides{int64_t(s)*3584,128,int64_t(s)*128,1};
}
inline bool disjoint(uintptr_t a,size_t an,uintptr_t b,size_t bn) {
    if(!an || !bn || a>UINTPTR_MAX-an || b>UINTPTR_MAX-bn)return false;
    return a+an<=b || b+bn<=a;
}
// Bounding spans of interleaved Q/K overlap, but their per-token rows do not.
inline bool qk_alias_safe(uintptr_t q,uintptr_t k,size_t b,size_t s,const Strides &qt,const Strides &kt) {
    if(!shape(b,s,28,128) || !input(s,qt) || !input(s,kt))return false;
    size_t qspan=((b*s-1)*size_t(qt[1])+3584)*2,kspan=((b*s-1)*size_t(kt[1])+3584)*2;
    if(disjoint(q,qspan,k,kspan))return true;
    return qt==kt && qt[1]==10752 && q<=UINTPTR_MAX-7168 && k==q+7168;
}
}
