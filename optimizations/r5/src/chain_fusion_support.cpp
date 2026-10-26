// Cross-layer residual is passed explicitly; no thread-local hidden-state cache.
#include <infinicore/tensor.hpp>
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
std::atomic<unsigned long long> fast{0},fallback{0},layers{0},pending_layers{0},snapshots{0};
bool flag(const char *name){const char*x=std::getenv(name);return x && !std::strcmp(x,"1");}
}
extern "C" unsigned long long iv_chain_fast_calls(){return fast.load();}
extern "C" unsigned long long iv_chain_fallback_calls(){return fallback.load();}
extern "C" unsigned long long iv_chain_rows(){return layers.load();}
extern "C" unsigned long long iv_chain_pending_layers(){return pending_layers.load();}
extern "C" unsigned long long iv_chain_snapshots(){return snapshots.load();}
bool iv_chain_begin(bool supported) {
    if(flag("IV_OPT_CHAIN") && flag("IV_OPT_FUSION") && supported && !infinicore::context::isGraphRecording()) {++fast;return true;}
    ++fallback;return false;
}
void iv_chain_layer(bool pending){++layers;if(pending)++pending_layers;}
void iv_chain_observe(infinicore::Tensor x) {
    const char *name=std::getenv("IV_CHAIN_VERIFY_DIR");if(!name || !*name)return;
    const auto dir=std::filesystem::canonical(name);
    if(dir.parent_path().parent_path()!=std::filesystem::path("/tmp") ||
       dir.parent_path().filename().string().find("infinivalidate-chain-")!=0 ||
       (dir.filename()!="baseline_hidden" && dir.filename()!="candidate_hidden") ||
       x->dtype()!=infinicore::DataType::BF16 || x->numel()==0 || x->numel()>(size_t(1)<<22))throw std::runtime_error("Invalid hidden-state capture scope");
    const auto index=snapshots.fetch_add(1);if(index>=96)throw std::runtime_error("Too many hidden-state captures");
    auto cpu=x->contiguous()->to(infinicore::Device::cpu());
    const auto path=dir/(std::to_string(index)+".bf16");
    int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(fd<0)throw std::runtime_error("Cannot create fresh hidden-state capture");
    size_t left=cpu->numel()*2;const auto *data=reinterpret_cast<const char*>(cpu->data());
    while(left){auto n=::write(fd,data,left);if(n<=0){::close(fd);throw std::runtime_error("Hidden-state capture failed");}data+=n;left-=size_t(n);}
    if(::close(fd))throw std::runtime_error("Hidden-state capture close failed");
}
