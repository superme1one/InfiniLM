// Request-local CPU metadata, inspired by SGLang's CPU/GPU length mirrors.
// No address cache, inferred lengths, global request state, or upstream code.
#include <infinicore/tensor.hpp>
#include <infinicore/context/context.hpp>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <optional>
namespace {
std::atomic<unsigned long long> fast{0},fallback{0},elements{0},copies{0},errors{0};
}
extern "C" unsigned long long iv_length_fast_calls(){return fast.load();}
extern "C" unsigned long long iv_length_fallback_calls(){return fallback.load();}
extern "C" unsigned long long iv_length_rows(){return elements.load();}
extern "C" unsigned long long iv_length_device_copies(){return copies.load();}
extern "C" unsigned long long iv_length_errors(){return errors.load();}

std::optional<infinicore::Tensor> iv_request_length_mirror(
    std::optional<infinicore::Tensor> lengths,bool static_cache) {
    using namespace infinicore;
    const char *flag=std::getenv("IV_OPT_LENGTH");
    if(!flag || std::strcmp(flag,"1") || !static_cache || !lengths.has_value() ||
       lengths.value()->dtype()!=DataType::I64 || lengths.value()->numel()==0 ||
       context::isGraphRecording()) {
        ++fallback;return lengths;
    }
    const auto &x=lengths.value();
    if(x->device().getType()!=Device::Type::CPU && x->device().getType()!=Device::Type::ILUVATAR) {
        ++fallback;return lengths;
    }
    try {
        // Retained only by this forward call. All layers receive this same tensor.
        auto cpu=x->to(Device::cpu());
        ++fast;elements+=x->numel();
        if(x->device().getType()!=Device::Type::CPU)++copies;
        return cpu;
    } catch(...) {
        // Preserve the original per-layer helper's exception/fallback semantics.
        ++errors;++fallback;return lengths;
    }
}
