#pragma once

#include "../config.hpp"

namespace PKAI {
    template<int_t _size>
    struct Layer {
        constexpr static int_t _is_layer_ = true;
        static constexpr bool _is_host = true;
        static constexpr bool _is_device = true;

        constexpr static int_t size = _size;
    };
}