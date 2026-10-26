// Only a RankWorker sampling call creates a Scope. Direct public model forwards
// retain full logits. Arithmetic remains the existing InfiniCore Linear/GEMM.
#include "last_logits_scope.hpp"
#include <infinicore/context/context.hpp>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
namespace {
thread_local iv_last_logits::Scope *current=nullptr;
std::atomic<unsigned long long> fast{0},fallback{0},rows{0},saved{0},snapshots{0};
bool enabled(){const char *v=std::getenv("IV_OPT_LAST_LOGITS");return v && !std::strcmp(v,"1");}
}
extern "C" unsigned long long iv_lastlogits_fast_calls(){return fast.load();}
extern "C" unsigned long long iv_lastlogits_fallback_calls(){return fallback.load();}
extern "C" unsigned long long iv_lastlogits_rows(){return rows.load();}
extern "C" unsigned long long iv_lastlogits_saved_rows(){return saved.load();}
extern "C" unsigned long long iv_lastlogits_snapshots(){return snapshots.load();}
extern "C" bool iv_lastlogits_context_eligible(){return current && current->eligible();}
namespace iv_last_logits {
Scope::Scope(std::optional<infinicore::Tensor> ids,std::optional<infinicore::Tensor> offsets,bool regular):previous_(current) {
    // Publish only after validation; an exception during construction must not
    // leave current pointing at an object whose constructor never completed.
    auto initialize=[&]() {
    if(!regular || !ids.has_value() || !offsets.has_value())return;
    const auto &x=ids.value();const auto &o=offsets.value();
    if(x->ndim()!=2 || x->shape()[0]==0 || x->shape()[0]>16 || x->shape()[1]==0 || x->shape()[1]>2048 ||
        x->dtype()!=infinicore::DataType::I64 || o->device().getType()!=infinicore::Device::Type::CPU ||
        o->dtype()!=infinicore::DataType::I64 || o->shape()!=infinicore::Shape{x->shape()[0]+1} || o->stride(0)!=1)return;
    batch_=x->shape()[0];sequence_=x->shape()[1];
    auto *values=reinterpret_cast<const int64_t*>(o->data());
    for(size_t i=0;i<=batch_;++i)if(values[i]!=int64_t(i*sequence_))return;
    regular_=true;
    const char *pad=std::getenv("IV_LOGITS_PAD_ROWS");
    if(pad && *pad){
        if(!std::strcmp(pad,"64"))pad_=64;
        else if(!std::strcmp(pad,"128"))pad_=128;
        else if(!std::strcmp(pad,"256"))pad_=256;
        else return;
    }
    eligible_=enabled() && sequence_>pad_ && sequence_%pad_==0 &&
        ((batch_==4 && sequence_==512) || (batch_==16 && sequence_==2048));
    };
    initialize();current=this;
}
Scope::~Scope(){current=previous_;}

infinicore::Tensor project(std::shared_ptr<infinicore::nn::Linear> head,infinicore::Tensor hidden) {
    using namespace infinicore;
    Scope *s=current;const auto shape=hidden->shape();const size_t width=shape.empty()?0:shape.back();
    if(!s || !s->eligible_ || shape!=Shape{s->batch_,s->sequence_,3584} || hidden->dtype()!=DataType::BF16 ||
       hidden->device().getType()!=Device::Type::ILUVATAR || context::isGraphRecording()) {
        ++fallback;if(width)rows+=hidden->numel()/width;return head->forward(hidden);
    }
    // Keep each selected row at the same row-within-tile position. Padding also
    // avoids sending a tiny M to a different generic GEMM implementation.
    auto input=Tensor::zeros({s->batch_,s->pad_,width},hidden->dtype(),hidden->device());
    input->narrow({{1,s->pad_-1,1}})->copy_from(hidden->narrow({{1,s->sequence_-1,1}}));
    auto projected=head->forward(input);
    auto selected=projected->narrow({{1,s->pad_-1,1}})->contiguous();
    ++fast;rows+=s->batch_*s->pad_;saved+=s->batch_*(s->sequence_-s->pad_);s->compact_=true;
    return selected;
}

void Scope::observe(infinicore::Tensor logits) const {
    const char *name=std::getenv("IV_LOGITS_VERIFY_DIR");if(!name || !*name)return;
    if(!regular_ || logits->ndim()!=3 || logits->shape()[0]!=batch_ || logits->shape()[1]!=(compact_?1:sequence_) ||
       logits->dtype()!=infinicore::DataType::BF16)throw std::runtime_error("Unexpected logits verification shape");
    const auto dir=std::filesystem::canonical(name);
    if(dir.parent_path().parent_path()!=std::filesystem::path("/tmp") || dir.parent_path().filename().string().find("infinivalidate-lastlogits-")!=0 ||
       (dir.filename()!="baseline_logits" && dir.filename()!="candidate_logits"))throw std::runtime_error("Invalid logits capture scope");
    auto selected=compact_?logits:logits->narrow({{1,sequence_-1,1}})->contiguous();
    if(selected->numel()==0 || selected->numel()>(size_t(1)<<23))throw std::runtime_error("Logits capture too large");
    auto index=snapshots.fetch_add(1);if(index>=96)throw std::runtime_error("Too many logits snapshots");
    auto cpu=selected->to(infinicore::Device::cpu());const auto path=dir/(std::to_string(index)+".bf16");
    int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);if(fd<0)throw std::runtime_error("Cannot create fresh logits snapshot");
    size_t left=cpu->numel()*2;const auto *data=reinterpret_cast<const char*>(cpu->data());
    while(left){auto n=::write(fd,data,left);if(n<=0){::close(fd);throw std::runtime_error("Logits snapshot write failed");}data+=n;left-=size_t(n);}
    if(::close(fd))throw std::runtime_error("Logits snapshot close failed");
}
}
