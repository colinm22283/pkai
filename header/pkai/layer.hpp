#pragma once

#include <pkai/common.hpp>

namespace PKAI {
    template<nsize_t _size>
    struct Layer {
        static constexpr bool is_layer = true;
        static constexpr nsize_t size = _size;
    };
}