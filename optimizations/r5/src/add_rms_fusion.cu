// Experimental out-of-place Add + RMSNorm for the audited TG200 backend.
// Architecture reference: vLLM residual norm fusion. Arithmetic is InfiniCore's:
// round the Add to BF16 BEFORE sumSquared; preserve the 2048-thread reduction.
// No upstream framework kernel was copied. Never replace installed libraries.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <infinicore/ops/add.hpp>
#include <infinicore/ops/rms_norm.hpp>
#include <infinicore/context/context.hpp>
#include "inference_candidate_memory_guard.hpp"
using cuda_bfloat16 = nv_bfloat16;
using cuda_bfloat162 = nv_bfloat162;
#include "infiniop/ops/add/cuda/kernel.cuh"
#include "infiniop/reduce/cuda/reduce.cuh"
#include "infiniop/ops/rms_norm/cuda/kernel.cuh"

namespace {
std::atomic<unsigned long long> fast{0}, fallback{0}, rows{0};
bool contiguous(infinicore::Tensor x) {
    const auto &s=x->shape(); const auto &t=x->strides();
    if(s.empty() || s.size()!=t.size())return false;
    size_t expected=1;
    for(size_t d=s.size();d>0;--d) {
        if(!s[d-1] || (s[d-1]>1 && (t[d-1]<=0 || size_t(t[d-1])!=expected)) ||
           !iv_memory_guard::multiply(expected,s[d-1],expected))return false;
    }
    return expected==x->numel();
}
bool block_supported(int device) {
    // Per-thread/device cache: do not query device properties on every layer.
    thread_local int cached_device=-1;
    thread_local bool supported=false;
    if(cached_device!=device) {
        cudaDeviceProp p{};
        auto rc=cudaGetDeviceProperties(&p,device);
        if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));
        supported=p.maxThreadsPerBlock==2048 && p.sharedMemPerBlock>=32768;
        cached_device=device;
    }
    return supported;
}
__global__ void iv_add_rms_rounded_2048(cuda_bfloat16 *y,cuda_bfloat16 *residual,
    const cuda_bfloat16 *a,const cuda_bfloat16 *b,const cuda_bfloat16 *w,size_t dim,float eps) {
    extern __shared__ unsigned short storage[];
    auto rounded=reinterpret_cast<cuda_bfloat16 *>(storage);
    const size_t offset=size_t(blockIdx.x)*dim;
    for(size_t i=threadIdx.x;i<dim;i+=2048) {
        auto v=::op::add::cuda::AddOp{}(a[offset+i],b[offset+i]);
        rounded[i]=v;residual[offset+i]=v;
    }
    __syncthreads();
    // Original RMSNorm body, with shared-memory input and zero shared row stride.
    rmsnormBlock<2048,float>(y,dim,dim,rounded,0,0,w,1,dim,eps);
}
}
extern "C" unsigned long long iv_fusion_fast_calls(){return fast.load();}
extern "C" unsigned long long iv_fusion_fallback_calls(){return fallback.load();}
extern "C" unsigned long long iv_fusion_rows(){return rows.load();}

// Fresh outputs eliminate cross-row aliasing and keep the original inputs intact.
std::pair<infinicore::Tensor,infinicore::Tensor> iv_add_rms_fusion(
    infinicore::Tensor a,infinicore::Tensor b,infinicore::Tensor weight,float epsilon) {
    using namespace infinicore;
    const char *flag=std::getenv("IV_OPT_FUSION");
    const auto d=a->device();const auto &shape=a->shape();
    const size_t dim=shape.empty()?0:shape.back();
    bool eligible=flag && !std::strcmp(flag,"1") && d.getType()==Device::Type::ILUVATAR &&
        a->dtype()==DataType::BF16 && b->dtype()==DataType::BF16 && weight->dtype()==DataType::BF16 &&
        b->device()==d && weight->device()==d && shape==b->shape() && (shape.size()==2 || shape.size()==3) &&
        dim && dim<=8192 && a->numel()<=(size_t(1)<<28) && weight->shape()==Shape{dim} &&
        contiguous(a) && contiguous(b) && contiguous(weight) && std::isfinite(epsilon) && epsilon>0 &&
        !context::isGraphRecording();
    if(eligible) {context::setDevice(d);eligible=block_supported(d.getIndex());}
    if(!eligible) {
        ++fallback;auto r=infinicore::op::add(a,b);return {infinicore::op::rms_norm(r,weight,epsilon),r};
    }
    auto y=Tensor::empty(shape,a->dtype(),d),r=Tensor::empty(shape,a->dtype(),d);
    size_t count=a->numel()/dim;
    iv_add_rms_rounded_2048<<<count,2048,dim*2,reinterpret_cast<cudaStream_t>(context::getStream())>>>(
        reinterpret_cast<cuda_bfloat16 *>(y->data()),reinterpret_cast<cuda_bfloat16 *>(r->data()),
        reinterpret_cast<const cuda_bfloat16 *>(a->data()),reinterpret_cast<const cuda_bfloat16 *>(b->data()),
        reinterpret_cast<const cuda_bfloat16 *>(weight->data()),dim,epsilon);
    auto rc=cudaGetLastError();if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));
    ++fast;rows+=count;return {y,r};
}
