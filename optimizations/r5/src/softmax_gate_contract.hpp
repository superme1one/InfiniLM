#pragma once
// Pure CPU fixtures and assertions shared by the GPU gate and its CPU tests.
#include "softmax_row_buckets.hpp"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace iv_softmax_gate {
constexpr size_t GUARD=32;
inline int checked_gpu(const char *value,const char *cuda,const char *ix,const char *monitor) {
    if(!value || !cuda || !ix || !monitor || !*monitor)throw std::invalid_argument("monitored GPU selection required");
    const std::string text(value);
    if(text.empty() || text.size()>2 || (text.size()>1 && text[0]=='0'))throw std::invalid_argument("noncanonical GPU index");
    int index=0;
    for(char c:text){if(c<'0' || c>'9')throw std::invalid_argument("invalid GPU index");index=index*10+c-'0';}
    if(index>63 || text!=cuda || text!=ix)throw std::invalid_argument("GPU visibility mismatch");
    return index;
}
enum class Pattern { Finite, SignedZero, Infinities, NaNPayloads, Extremes, NegInf, PosInf };
inline const char *pattern_name(Pattern p) {
    static const char *names[]={"finite","signed_zero","infinities","nan_payloads","extremes","all_negative_inf","all_positive_inf"};
    return names[static_cast<size_t>(p)];
}
struct Spec {
    size_t batch,height,width;
    Pattern pattern=Pattern::Finite;
    bool inplace=false;
    size_t row_extra=0,effective=0;
    size_t pitch()const{return width+row_extra;}
    size_t storage_size()const{return 2*GUARD+batch*height*pitch();}
    size_t numel()const{return batch*height*width;}
    bool eligible()const{return row_extra==0 && effective==0;}
};
inline void print_spec(const char *prefix,size_t id,const Spec &s) {
    std::printf("%s {\"case_id\":%zu,\"batch\":%zu,\"height\":%zu,\"width\":%zu,\"pattern\":\"%s\",\"inplace\":%s,\"row_extra\":%zu,\"effective\":%zu,\"eligible\":%s}\n",prefix,id,s.batch,s.height,s.width,pattern_name(s.pattern),s.inplace?"true":"false",s.row_extra,s.effective,s.eligible()?"true":"false");
}
inline std::vector<Spec> plan() {
    std::vector<Spec> out;
    const std::vector<Spec> shapes={{1,1,1},{1,1,65},{1,1,257},{1,3,65},{1,64,64},{1,65,65},
        {2,3,257},{1,257,257},{1,513,513},{1,1025,1025},{1,2048,2048},{1,1,2080}};
    for(auto s:shapes)for(bool inplace:{false,true}){s.inplace=inplace;out.push_back(s);}
    for(auto p:{Pattern::SignedZero,Pattern::Infinities,Pattern::NaNPayloads,Pattern::Extremes,Pattern::NegInf,Pattern::PosInf})
        for(auto s:std::vector<Spec>{{1,3,65},{1,65,65},{1,257,257},{1,1,2080}})
            for(bool inplace:{false,true}){s.pattern=p;s.inplace=inplace;out.push_back(s);}
    for(auto s:std::vector<Spec>{{2,3,65},{1,65,65}})for(bool inplace:{false,true}) {
        s.inplace=inplace;s.row_extra=8;s.effective=0;out.push_back(s);
        s.row_extra=0;s.effective=s.width-1;out.push_back(s);
    }
    return out;
}
inline uint16_t bf16(float value) {
    uint32_t b;std::memcpy(&b,&value,4);return static_cast<uint16_t>((b+0x7fff+((b>>16)&1))>>16);
}
inline uint16_t input_value(Pattern pattern,size_t index,uint32_t &state) {
    state^=state<<13;state^=state>>17;state^=state<<5;
    switch(pattern) {
        case Pattern::SignedZero:return index%2?0x8000:0;
        case Pattern::Infinities:{const uint16_t v[]={0x7f80,0xff80,0,0x3f80};return v[index%4];}
        case Pattern::NaNPayloads:{const uint16_t v[]={0x7fc1,0x7fff,0xffc2,0x3f80,0};return v[index%5];}
        case Pattern::Extremes:{const uint16_t v[]={0x7f7f,0xff7f,1,0x8001,0x0080,0x8080,0x3f80,0xbf80};return v[index%8];}
        case Pattern::NegInf:return 0xff80;
        case Pattern::PosInf:return 0x7f80;
        default:return bf16((static_cast<int>(state%32769)-16384)/1024.0f);
    }
}
inline std::vector<uint16_t> input_storage(const Spec &s,uint32_t seed) {
    std::vector<uint16_t> out(s.storage_size(),0x6b6b);size_t logical=0;
    for(size_t row=0;row<s.batch*s.height;++row)for(size_t col=0;col<s.width;++col)
        out[GUARD+row*s.pitch()+col]=input_value(s.pattern,logical++,seed);
    return out;
}
inline std::vector<uint16_t> extract_guarded(const std::vector<uint16_t> &raw,const std::vector<uint16_t> &initial,const Spec &s) {
    if(raw.size()!=s.storage_size() || initial.size()!=raw.size())throw std::runtime_error("storage extent mismatch");
    std::vector<uint16_t> values;values.reserve(s.numel());
    for(size_t i=0;i<raw.size();++i) {
        const bool inside=i>=GUARD && i<raw.size()-GUARD;
        const bool logical=inside && ((i-GUARD)%s.pitch()<s.width);
        if(logical)values.push_back(raw[i]);
        else if(raw[i]!=initial[i])throw std::runtime_error("guard or row-padding overwritten");
    }
    return values;
}
inline bool mask_is_zero(const std::vector<uint16_t> &values,const Spec &s) {
    if(values.size()!=s.numel())return false;
    for(size_t b=0;b<s.batch;++b)for(size_t row=0;row<s.height;++row) {
        size_t allowed=s.width-s.height+1+row;
        if(s.effective>0 && s.effective<=s.width)allowed=std::min(allowed,s.effective);
        for(size_t col=allowed;col<s.width;++col)if(values[(b*s.height+row)*s.width+col]!=0)return false;
    }
    return true;
}
struct Stats {
    unsigned long long fast=0,fallback=0,clone=0,kernels=0,reduced=0;
};
inline Stats delta(Stats a,Stats b) {return {b.fast-a.fast,b.fallback-a.fallback,b.clone-a.clone,b.kernels-a.kernels,b.reduced-a.reduced};}
inline bool equal(Stats a,Stats b){return a.fast==b.fast && a.fallback==b.fallback && a.clone==b.clone && a.kernels==b.kernels && a.reduced==b.reduced;}
inline Stats expected(const Spec &s,const std::string &mode) {
    if(mode!="0" && mode!="2048" && mode!="1")throw std::invalid_argument("unknown mode");
    if(mode=="0" || !s.eligible())return {0,1,0,0,0};
    if(mode=="2048")return {0,0,1,1,0};
    const auto groups=iv_softmax::plan(s.height,s.width);unsigned long long reduced=0;
    for(auto g:groups)if(g.block<2048)reduced+=g.rows*s.batch;
    return {1,0,0,groups.size(),reduced};
}
inline size_t unequal(const std::vector<uint16_t> &a,const std::vector<uint16_t> &b) {
    if(a.size()!=b.size())throw std::runtime_error("logical output extent mismatch");
    size_t count=0;for(size_t i=0;i<a.size();++i)count+=a[i]!=b[i];return count;
}
}
