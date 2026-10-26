// Q out-of-place and K in-place in one launch; original BF16 device body.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>
#include "rope_pair_launch.hpp"
using cuda_bfloat16=nv_bfloat16;
using cuda_bfloat162=nv_bfloat162;
#include "infiniop/ops/rope/cuda/kernel.cuh"
template<bool GPTJ>
__global__ void pair_kernel(cuda_bfloat16 *y,const cuda_bfloat16 *q,cuda_bfloat16 *k,
    const int64_t *pos,const cuda_bfloat16 *sin,const cuda_bfloat16 *cos,
    IvRopePairStrides yt,IvRopePairStrides qt,IvRopePairStrides kt){
    ropeThreadPerItemBlock<GPTJ>(y,q,pos,sin,cos,64,0,false,true,yt.b,yt.s,yt.h,qt.b,qt.s,qt.h);
    ropeThreadPerItemBlock<GPTJ>(k,k,pos,sin,cos,64,0,false,true,kt.b,kt.s,kt.h,kt.b,kt.s,kt.h);
}
extern "C" int iv_rope_pair_launch(void *y,const void *q,void *k,const void *pos,const void *sin,const void *cos,
    size_t batch,size_t seq,bool gptj,IvRopePairStrides yt,IvRopePairStrides qt,IvRopePairStrides kt,void *stream){
    dim3 grid(seq,28,batch);auto st=reinterpret_cast<cudaStream_t>(stream);
    auto yp=static_cast<cuda_bfloat16*>(y),kp=static_cast<cuda_bfloat16*>(k);
    auto qp=static_cast<const cuda_bfloat16*>(q),sp=static_cast<const cuda_bfloat16*>(sin),cp=static_cast<const cuda_bfloat16*>(cos);
    auto pp=static_cast<const int64_t*>(pos);
    if(gptj)pair_kernel<true><<<grid,64,0,st>>>(yp,qp,kp,pp,sp,cp,yt,qt,kt);
    else pair_kernel<false><<<grid,64,0,st>>>(yp,qp,kp,pp,sp,cp,yt,qt,kt);
    return int(cudaGetLastError());
}
