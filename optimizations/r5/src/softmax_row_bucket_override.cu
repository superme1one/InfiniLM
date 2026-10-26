// PENDING GPU VALIDATION. Process-local opt-in; never replace installed libraries.
// Reuses the exact inspected server kernel, including BF16 exp and division.
// CPU sum-tree equivalence is limited to positive finite exponent results; it
// does NOT establish whole-kernel NaN/Inf behavior or GPU/model correctness.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <infinicore/ops/causal_softmax.hpp>
#include <infinicore/context/context.hpp>
#include <infiniop/ops/causal_softmax.h>
using cuda_bfloat16=nv_bfloat16;
using cuda_bfloat162=nv_bfloat162;
#include "infiniop/reduce/cuda/reduce.cuh"
#include "infiniop/ops/causal_softmax/cuda/kernel.cuh"
#include "softmax_row_buckets.hpp"

namespace {
using Fn=void(*)(infinicore::Tensor,infinicore::Tensor);
std::atomic<unsigned long long> fast_calls{0},fallback_calls{0},clone_calls{0},launches{0},reduced_rows{0};
Fn original() {
    static auto fn=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"_ZN10infinicore2op15causal_softmax_ENS_6TensorES1_"));
    if(!fn)throw std::runtime_error("Cannot resolve original causal_softmax_");
    return fn;
}
void check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
template<unsigned B> __global__ void iv_causal_softmax_rows(
    cuda_bfloat16 *out,const cuda_bfloat16 *input,size_t batch,size_t shifted_height,size_t width,
    ptrdiff_t out_batch,ptrdiff_t out_row,ptrdiff_t in_batch,ptrdiff_t in_row) {
    causalSoftmaxKernel<B,cuda_bfloat16,float>(out,input,batch,shifted_height,width,0,
        out_batch,out_row,in_batch,in_row);
}
template<unsigned B>void launch(infinicore::Tensor out,infinicore::Tensor input,
                              size_t start,size_t rows,cudaStream_t stream) {
    const size_t batch=out->size(0),height=out->size(1),width=out->size(2);
    auto y=reinterpret_cast<cuda_bfloat16*>(out->data())+start*out->stride(1);
    auto x=reinterpret_cast<const cuda_bfloat16*>(input->data())+start*input->stride(1);
    iv_causal_softmax_rows<B><<<dim3(rows,batch),B,0,stream>>>(y,x,batch,height-start,width,
        out->stride(0),out->stride(1),input->stride(0),input->stride(1));
    check(cudaGetLastError());++launches;
    if(B<2048)reduced_rows+=rows*batch;
}
}
extern "C" unsigned long long iv_softmax_fast_calls(){return fast_calls.load();}
extern "C" unsigned long long iv_softmax_fallback_calls(){return fallback_calls.load();}
extern "C" unsigned long long iv_softmax_clone_calls(){return clone_calls.load();}
extern "C" unsigned long long iv_softmax_kernel_launches(){return launches.load();}
extern "C" unsigned long long iv_softmax_reduced_rows(){return reduced_rows.load();}

namespace infinicore::op {
void causal_softmax_(Tensor out,Tensor input) {
    const char *mode=std::getenv("IV_OPT_SOFTMAX");
    const bool bucket=mode && std::strcmp(mode,"1")==0;
    const bool clone=mode && std::strcmp(mode,"2048")==0; // arithmetic/source contract diagnostic
    const auto device=out->device();
    if((!bucket&&!clone) || device.getType()!=Device::Type::ILUVATAR ||
       input->device()!=device || out->dtype()!=DataType::BF16 || input->dtype()!=out->dtype() ||
       out->ndim()!=3 || out->shape()!=input->shape() || !out->is_contiguous() || !input->is_contiguous() ||
       !out->size(0) || out->size(0)>65535 || !out->size(1) || out->size(1)>4096 ||
       out->size(2)<out->size(1) || out->size(2)>65536 || context::isGraphRecording() ||
       infiniopGetCausalSoftmaxEffectiveLength()!=0) {
        ++fallback_calls;original()(out,input);return;
    }
    const auto a=reinterpret_cast<uintptr_t>(out->data()),b=reinterpret_cast<uintptr_t>(input->data());
    // Same-pointer in-place is allowed; reject every partial overlap.
    const auto distance=a>b?a-b:b-a;
    if(a!=b && distance<out->nbytes()){++fallback_calls;original()(out,input);return;}
    context::setDevice(device);
    static thread_local int checked_device=-1,max_threads=0;
    if(checked_device!=device.getIndex()) {
        check(cudaDeviceGetAttribute(&max_threads,cudaDevAttrMaxThreadsPerBlock,device.getIndex()));
        checked_device=device.getIndex();
    }
    if(max_threads!=2048){++fallback_calls;original()(out,input);return;}
    auto stream=reinterpret_cast<cudaStream_t>(context::getStream());
    if(clone){launch<2048>(out,input,0,out->size(1),stream);++clone_calls;return;}
    for(auto group:iv_softmax::plan(out->size(1),out->size(2))) {
        switch(group.block) {
            case 256:launch<256>(out,input,group.start,group.rows,stream);break;
            case 512:launch<512>(out,input,group.start,group.rows,stream);break;
            case 1024:launch<1024>(out,input,group.start,group.rows,stream);break;
            default:launch<2048>(out,input,group.start,group.rows,stream);break;
        }
    }
    ++fast_calls;
}
}
