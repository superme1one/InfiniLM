// Isolated public API extension: use this RoPE instance's original cache.
#include <infinicore/nn/rope.hpp>
#include <atomic>
void iv_rope_pair(infinicore::Tensor, infinicore::Tensor, infinicore::Tensor,
                  infinicore::Tensor, infinicore::Tensor, infinicore::Tensor,
                  infinicore::nn::RoPE::Algo);
namespace { std::atomic<unsigned long long> api_calls{0}; }
extern "C" unsigned long long iv_rope_pair_api_calls() { return api_calls.load(); }
void infinicore::nn::RoPE::forward_qk(const Tensor &q_out, const Tensor &q,
                                    const Tensor &k, const Tensor &pos) const {
    iv_rope_pair(q_out, q, k, pos, sin_cache_, cos_cache_, algo_);
    ++api_calls;
}
