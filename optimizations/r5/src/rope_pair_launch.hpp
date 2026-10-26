#pragma once
#include <cstddef>
struct IvRopePairStrides{ptrdiff_t b,s,h;};
extern "C" int iv_rope_pair_launch(void *y,const void *q,void *k,const void *pos,const void *sin,const void *cos,
    size_t batch,size_t seq,bool gptj,IvRopePairStrides yt,IvRopePairStrides qt,IvRopePairStrides kt,void *stream);
