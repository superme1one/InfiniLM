// Experimental, process-local Add specialization. Never replace installed libs.
// Same BF16 AddOp as the read-only backend; no vector rounding substitution.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include "inference_candidate_memory_guard.hpp"
#include <infinicore/ops/add.hpp>
#include <infinicore/context/context.hpp>
using cuda_bfloat16 = nv_bfloat16;
using cuda_bfloat162 = nv_bfloat162;
#include "infiniop/ops/add/cuda/kernel.cuh"

namespace {
using Fn = void (*)(infinicore::Tensor, infinicore::Tensor, infinicore::Tensor);
std::atomic<unsigned long long> fast_calls{0}, fallback_calls{0}, elements{0};
Fn original() {
    static auto fn = reinterpret_cast<Fn>(dlsym(RTLD_NEXT,
        "_ZN10infinicore2op3Add7executeENS_6TensorES2_S2_"));
    if (!fn) throw std::runtime_error("Cannot resolve original Add::execute");
    return fn;
}
bool contiguous(infinicore::Tensor x) {
    const auto &shape=x->shape(); const auto &strides=x->strides();
    if (shape.empty() || shape.size()!=strides.size()) return false;
    size_t expected=1;
    for(size_t d=shape.size(); d>0; --d) {
        if (!shape[d-1] || (shape[d-1]>1 &&
            (strides[d-1]<=0 || size_t(strides[d-1])!=expected))) return false;
        if (!iv_memory_guard::multiply(expected,shape[d-1],expected)) return false;
    }
    return expected==x->numel();
}
__global__ void iv_add_contiguous(cuda_bfloat16 *c, const cuda_bfloat16 *a,
                                  const cuda_bfloat16 *b, size_t count) {
    const size_t first=size_t(blockIdx.x)*1024+threadIdx.x;
    #pragma unroll
    for(unsigned lane=0;lane<4;++lane) {
        const size_t i=first+lane*256;
        if(i<count) c[i]=::op::add::cuda::AddOp{}(a[i],b[i]);
    }
}
}
extern "C" unsigned long long iv_add_fast_calls() { return fast_calls.load(); }
extern "C" unsigned long long iv_add_fallback_calls() { return fallback_calls.load(); }
extern "C" unsigned long long iv_add_rows() { return elements.load(); }
namespace infinicore::op {
void Add::execute(Tensor c, Tensor a, Tensor b) {
    const char *enabled=std::getenv("IV_OPT_ADD");
    const auto device=c->device();
    const size_t n=c->numel();
    if (!enabled || std::strcmp(enabled,"1") ||
        (device.getType()!=Device::Type::ILUVATAR && device.getType()!=Device::Type::NVIDIA) ||
        c->dtype()!=DataType::BF16 || a->dtype()!=c->dtype() || b->dtype()!=c->dtype() ||
        a->device()!=device || b->device()!=device || c->shape()!=a->shape() || c->shape()!=b->shape() ||
        !n || n>(size_t(1)<<28) || !contiguous(c) || !contiguous(a) || !contiguous(b) ||
        context::isGraphRecording() ||
        !iv_memory_guard::elementwise_input_allowed(reinterpret_cast<uintptr_t>(c->data()),reinterpret_cast<uintptr_t>(a->data()),1,n,n,2) ||
        !iv_memory_guard::elementwise_input_allowed(reinterpret_cast<uintptr_t>(c->data()),reinterpret_cast<uintptr_t>(b->data()),1,n,n,2)) {
        ++fallback_calls; original()(c,a,b); return;
    }
    context::setDevice(device);
    iv_add_contiguous<<<(n+1023)/1024,256,0,reinterpret_cast<cudaStream_t>(context::getStream())>>>(
        reinterpret_cast<cuda_bfloat16 *>(c->data()),reinterpret_cast<const cuda_bfloat16 *>(a->data()),
        reinterpret_cast<const cuda_bfloat16 *>(b->data()),n);
    const auto rc=cudaGetLastError();
    if(rc!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
    ++fast_calls; elements+=n;
}
}
