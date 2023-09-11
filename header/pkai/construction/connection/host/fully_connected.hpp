#pragma once

#include "../../../config.hpp"

namespace PKAI::Connection::Host {
    struct FullyConnected {
        static constexpr bool _is_connection_ = true;
        static constexpr bool _is_host = true;
        static constexpr bool _is_device = false;

        template<int_t from_size, int_t to_size>
        static constexpr int_t allocation_size = from_size * to_size;
    };
}