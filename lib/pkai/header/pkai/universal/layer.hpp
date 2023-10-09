#pragma once

#ifndef PKAI_HOST
#ifndef PKAI_DEVICE
#error PKAI_HOST or PKAI_DEVICE must be defined
#endif
#endif

#include "config.hpp"

namespace PKAI {
    template<int_t _size>
    struct Layer {
        constexpr static int_t _is_layer_ = true;
        static constexpr bool _is_host = true;
        static constexpr bool _is_device = true;

        constexpr static int_t size = _size;
    };
}