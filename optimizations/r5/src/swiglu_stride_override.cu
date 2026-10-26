// Experimental row-stride specialization. Original arithmetic functor is reused
// verbatim from the read-only source tree, including BF16 intermediate rounding.
// Current source adds CPU-reviewed alias eligibility; GPU revalidation is pending.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <type_traits>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include "inference_candidate_memory_guard.hpp"
#include <infinicore/ops/swiglu.hpp>
#include <infinicore/context/context.hpp>
using cuda_bfloat16 = nv_bfloat16;
using cuda_bfloat162 = nv_bfloat162;
#include "infiniop/ops/swiglu/cuda/kernel.cuh"

namespace {
using Fn = void (*)(infinicore::Tensor, infinicore::Tensor, infinicore::Tensor);
std::atomic<unsigned long long> fast_calls{0}, fallback_calls{0}, elements{0};
std::atomic<unsigned long long> memory_fallback_calls{0};
Fn original() {
    static auto fn = reinterpret_cast<Fn>(dlsym(RTLD_NEXT,
        "_ZN10infinicore2op7swiglu_ENS_6TensorES1_S1_"));
    if (!fn) throw std::runtime_error("Cannot resolve original swiglu_");
    return fn;
}
// Flatten only a proven regular leading-dimension layout. No broadcast assumption.
bool regular(infinicore::Tensor x, size_t cols, ptrdiff_t &row_stride) {
    const auto &shape = x->shape(); const auto &strides = x->strides();
    if (shape.empty() || strides.back()!=1 || shape.back()!=cols) return false;
    row_stride = shape.size()==1 ? cols : strides[shape.size()-2];
    if (row_stride < static_cast<ptrdiff_t>(cols)) return false;
    for (size_t i = shape.size(); i > 2; --i)
        if (!iv_memory_guard::merge_stride_matches(strides[i-3],strides[i-2],shape[i-2])) return false;
    return true;
}
__global__ void iv_swiglu_row_stride(cuda_bfloat16 *c, const cuda_bfloat16 *up,
                                    const cuda_bfloat16 *gate, size_t cols,
                                    ptrdiff_t up_stride, ptrdiff_t gate_stride) {
    const size_t row = blockIdx.y;
    const size_t first = size_t(blockIdx.x)*1024 + threadIdx.x;
    #pragma unroll
    for (unsigned lane=0; lane<4; ++lane) {
        const size_t col = first + lane*256;
        if (col<cols)
            c[row*cols+col] = ::op::swiglu::cuda::SwiGLUOp{}(up[row*up_stride+col],gate[row*gate_stride+col]);
    }
}
}
extern "C" unsigned long long iv_swiglu_fast_calls() { return fast_calls.load(); }
extern "C" unsigned long long iv_swiglu_fallback_calls() { return fallback_calls.load(); }
extern "C" unsigned long long iv_swiglu_rows() { return elements.load(); } // generic dispatch-stat compatibility
extern "C" unsigned long long iv_swiglu_memory_fallback_calls() { return memory_fallback_calls.load(); }

namespace infinicore::op {
void swiglu_(Tensor c, Tensor up, Tensor gate) {
    const char *enabled=std::getenv("IV_OPT_SWIGLU");
    const auto device=c->device();
    ptrdiff_t us=0,gs=0;
    const size_t cols=c->ndim() ? c->shape().back() : 0;
    const size_t rows=cols ? c->numel()/cols : 0;
    if (!enabled || enabled[0]!='1' ||
        (device.getType()!=Device::Type::ILUVATAR && device.getType()!=Device::Type::NVIDIA) ||
        c->dtype()!=DataType::BF16 || up->dtype()!=c->dtype() || gate->dtype()!=c->dtype() ||
        up->device()!=device || gate->device()!=device ||
        c->shape()!=up->shape() || c->shape()!=gate->shape() || !c->is_contiguous() ||
        cols==0 || cols>1048576 || rows==0 || rows>65535 || !regular(up,cols,us) || !regular(gate,cols,gs) ||
        // Installed InfiniOp misclassifies [r,1]/[2,1] as contiguous. Preserve
        // legacy behavior for this unsupported corner; the separate descriptor
        // fix needs its own correctness review. Never claim these as fast cases.
        (cols==1 && rows>1 && (!up->is_contiguous() || !gate->is_contiguous())) ||
        context::isGraphRecording()) {
        ++fallback_calls; original()(c,up,gate); return;
    }
    const auto out_address=reinterpret_cast<uintptr_t>(c->data());
    if (!iv_memory_guard::elementwise_input_allowed(out_address,reinterpret_cast<uintptr_t>(up->data()),rows,cols,us,2) ||
        !iv_memory_guard::elementwise_input_allowed(out_address,reinterpret_cast<uintptr_t>(gate->data()),rows,cols,gs,2)) {
        ++memory_fallback_calls; ++fallback_calls;
        original()(c,up,gate); return;
    }
    context::setDevice(device);
    auto stream=reinterpret_cast<cudaStream_t>(context::getStream());
    iv_swiglu_row_stride<<<dim3((cols+1023)/1024,rows),256,0,stream>>>(
        reinterpret_cast<cuda_bfloat16 *>(c->data()),
        reinterpret_cast<const cuda_bfloat16 *>(up->data()),
        reinterpret_cast<const cuda_bfloat16 *>(gate->data()),cols,us,gs);
    const auto rc=cudaGetLastError();
    if (rc!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
    ++fast_calls; elements+=c->numel();
}
}
