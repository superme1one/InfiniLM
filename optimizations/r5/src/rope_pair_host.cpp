// Host-only framework glue: keep nn/spdlog headers out of the device compiler.
#include <cuda_runtime.h>
#include <infinicore/ops/rope.hpp>
#include <infinicore/context/context.hpp>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "rope_pair_guard.hpp"
#include "rope_pair_launch.hpp"
using namespace infinicore;
namespace {
std::atomic<unsigned long long> fast{0},fallback{0},launches{0};
using Strides=iv_rope_pair_guard::Strides;
Strides strides(Tensor t){auto s=t->strides();return {s[0],s[1],s[2],s[3]};}
IvRopePairStrides device_strides(Tensor t){auto s=t->strides();return {s[0],s[1],s[2]};}
bool eligible(Tensor y,Tensor q,Tensor k,Tensor pos,Tensor sin,Tensor cos,nn::RoPE::Algo algo){
    using namespace iv_rope_pair_guard;
    auto d=q->device();auto sh=q->shape();
    if(d.getType()!=Device::Type::ILUVATAR || sh.size()!=4 || !shape(sh[0],sh[1],sh[2],sh[3]) || y->shape()!=sh || k->shape()!=sh || context::isGraphRecording())return false;
    if(algo!=nn::RoPE::Algo::GPT_NEOX && algo!=nn::RoPE::Algo::GPT_J)return false;
    for(auto t:{y,q,k,sin,cos})if(t->dtype()!=DataType::BF16 || t->device()!=d)return false;
    if(pos->device()!=d || pos->dtype()!=DataType::I64 || pos->shape()!=Shape{sh[1]} || pos->strides()!=std::vector<Stride>{1})return false;
    auto ss=sin->shape();
    // Position range is the same caller contract as the original RoPE API;
    // do not add a synchronizing D2H readback. The gate supplies checked values.
    if(ss.size()!=2 || ss[1]!=64 || ss[0]<sh[1] || ss[0]>131072 || cos->shape()!=ss || sin->strides()!=std::vector<Stride>{64,1} || cos->strides()!=std::vector<Stride>{64,1})return false;
    auto qt=strides(q),kt=strides(k),yt=strides(y);
    if(!input(sh[1],qt)||!input(sh[1],kt)||!output(sh[1],yt))return false;
    uintptr_t qa=reinterpret_cast<uintptr_t>(q->data()),ka=reinterpret_cast<uintptr_t>(k->data()),ya=reinterpret_cast<uintptr_t>(y->data());
    size_t qn=((sh[0]*sh[1]-1)*size_t(qt[1])+3584)*2,kn=((sh[0]*sh[1]-1)*size_t(kt[1])+3584)*2,yn=q->numel()*2;
    if(!qk_alias_safe(qa,ka,sh[0],sh[1],qt,kt) || !disjoint(ya,yn,qa,qn) || !disjoint(ya,yn,ka,kn))return false;
    for(auto t:{pos,sin,cos}){
        uintptr_t ta=reinterpret_cast<uintptr_t>(t->data());size_t bytes=t->numel()*(t->dtype()==DataType::I64?8:2);
        if(!disjoint(ya,yn,ta,bytes)||!disjoint(ka,kn,ta,bytes))return false;
    }
    return true;
}
}
extern "C" unsigned long long iv_rope_pair_fast(){return fast.load();}
extern "C" unsigned long long iv_rope_pair_fallback(){return fallback.load();}
extern "C" unsigned long long iv_rope_pair_launches(){return launches.load();}
void iv_rope_pair(Tensor y,Tensor q,Tensor k,Tensor pos,Tensor sin,Tensor cos,nn::RoPE::Algo algo){
    const char *flag=std::getenv("IV_OPT_ROPE_PAIR");
    if(!flag || std::strcmp(flag,"1") || !eligible(y,q,k,pos,sin,cos,algo)){
        ++fallback;op::rope_(y,q,pos,sin,cos,algo);op::rope_(k,k,pos,sin,cos,algo);launches+=2;return;
    }
    context::setDevice(q->device());auto sh=q->shape();
    auto rc=static_cast<cudaError_t>(iv_rope_pair_launch(y->data(),q->data(),k->data(),pos->data(),sin->data(),cos->data(),
        sh[0],sh[1],algo==nn::RoPE::Algo::GPT_J,device_strides(y),device_strides(q),device_strides(k),context::getStream()));
    if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));
    ++fast;++launches;
}
