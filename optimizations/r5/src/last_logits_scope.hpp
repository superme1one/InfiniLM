#pragma once
#include <infinicore/tensor.hpp>
#include <infinicore/nn/linear.hpp>
#include <optional>
namespace iv_last_logits {
// Stack-owned, nestable worker-call context. Not a cache of tensor addresses.
class Scope {
public:
    Scope(std::optional<infinicore::Tensor> input_ids,
          std::optional<infinicore::Tensor> offsets,bool regular);
    ~Scope();
    Scope(const Scope&)=delete;
    Scope& operator=(const Scope&)=delete;
    bool compact() const { return compact_; }
    bool eligible() const { return eligible_; }
    void observe(infinicore::Tensor logits) const;
private:
    friend infinicore::Tensor project(std::shared_ptr<infinicore::nn::Linear>,infinicore::Tensor);
    Scope *previous_=nullptr;
    size_t batch_=0,sequence_=0,pad_=64;
    bool regular_=false,eligible_=false,compact_=false;
};
infinicore::Tensor project(std::shared_ptr<infinicore::nn::Linear>,infinicore::Tensor);
}
