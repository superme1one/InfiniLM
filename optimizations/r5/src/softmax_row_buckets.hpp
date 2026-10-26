#pragma once
// Host-only launch planning. Does not initialize, query, or execute a GPU.
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace iv_softmax {
struct Bucket { size_t start, rows; unsigned block; };
inline std::vector<Bucket> plan(size_t height,size_t width) {
    if(!height || height>4096 || width<height || width>65536)
        throw std::invalid_argument("bounded softmax shape requires 1<=height<=4096 and height<=width<=65536");
    const size_t first_allowed=width-height+1;
    size_t start=0;std::vector<Bucket> result;
    for(unsigned limit:{256u,512u,1024u}) {
        if(first_allowed>limit)continue;
        const size_t end=std::min(height,size_t(limit)-first_allowed+1);
        if(end>start) {result.push_back({start,end-start,limit});start=end;}
    }
    if(start<height)result.push_back({start,height-start,2048});
    return result;
}
}
