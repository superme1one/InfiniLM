#pragma once
// Pure host arithmetic. Never dereferences a device address or queries a runtime.
// Conservative eligibility checks, NOT a promise that fallback supports aliasing.
#include <cstddef>
#include <cstdint>
#include <limits>

namespace iv_memory_guard {
struct Span {
    uintptr_t first=0,last=0; // Inclusive: a final byte at UINTPTR_MAX is representable.
    bool valid=false;
};
inline bool multiply(size_t a,size_t b,size_t &out) {
    if(b && a>std::numeric_limits<size_t>::max()/b)return false;
    out=a*b;return true;
}
inline bool merge_stride_matches(ptrdiff_t outer,ptrdiff_t inner,size_t extent) {
    if(outer<=0 || inner<=0 || extent==0 ||
       extent>static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()/inner))return false;
    return outer==inner*static_cast<ptrdiff_t>(extent);
}
inline Span rows(uintptr_t base,size_t count,size_t cols,size_t stride,size_t element_bytes) {
    if(!base || !count || !cols || !element_bytes || stride<cols)return {};
    size_t leading=0,bytes=0;
    if(!multiply(count-1,stride,leading) || cols>std::numeric_limits<size_t>::max()-leading ||
       !multiply(leading+cols,element_bytes,bytes) || !bytes ||
       bytes>static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) ||
       bytes-1>std::numeric_limits<uintptr_t>::max()-base)return {};
    return {base,base+(bytes-1),true};
}
inline bool disjoint(Span a,Span b) {
    return a.valid && b.valid && (a.last<b.first || b.last<a.first);
}
inline bool gather_allowed(uintptr_t output,uintptr_t weight,size_t count,size_t vocab,
                           size_t cols,size_t element_bytes) {
    // Any overlap is unsupported, even apparently identity indices: gather rows
    // execute independently, and another row may still read the overwritten data.
    return disjoint(rows(output,count,cols,cols,element_bytes),
                    rows(weight,vocab,cols,cols,element_bytes));
}
inline bool elementwise_input_allowed(uintptr_t output,uintptr_t input,size_t count,
                                     size_t cols,size_t input_stride,size_t element_bytes) {
    if(!element_bytes || output%element_bytes || input%element_bytes)return false;
    const auto out=rows(output,count,cols,cols,element_bytes);
    const auto in=rows(input,count,cols,input_stride,element_bytes);
    if(!out.valid || !in.valid)return false;
    if(disjoint(out,in))return true;
    // Same base is not enough for a pitched view: every output element must read
    // exactly its own input element. Padding-hole overlap may fall back needlessly;
    // conservatism is intentional until a separate alias contract is validated.
    return output==input && (count==1 || input_stride==cols);
}
}
