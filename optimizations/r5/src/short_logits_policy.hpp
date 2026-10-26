#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
namespace iv_short_policy {
enum class Scheme { Full, LegacyPadded, Compact };
inline bool valid_offsets(size_t batch,size_t sequence,const int64_t *offsets,size_t count,bool regular) {
    if(!regular||!offsets||batch==0||batch>16||sequence==0||sequence>2048||count!=batch+1)return false;
    for(size_t i=0;i<count;++i)if(offsets[i]!=int64_t(i*sequence))return false;
    return true;
}
inline Scheme select(size_t batch,size_t sequence,size_t pad,bool enabled,bool short_enabled,bool context_valid) {
    if(!enabled||!context_valid||(pad!=64&&pad!=128&&pad!=256))return Scheme::Full;
    if(short_enabled&&(batch==4||batch==16)&&sequence>=2&&sequence<=64)return Scheme::Compact;
    if(sequence>pad&&sequence%pad==0&&((batch==4&&sequence==512)||(batch==16&&sequence==2048)))return Scheme::LegacyPadded;
    return Scheme::Full;
}
}
