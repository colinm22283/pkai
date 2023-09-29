#pragma once

#include "../../../config.hpp"

namespace PKAI::Connection::Host {
    struct FullyConnected {
        static constexpr bool _is_connection_ = true;
        static constexpr bool _is_host = true;
        static constexpr bool _is_device = false;

        template<typename FloatType, int_t from_size, int_t to_size>
        struct Config {
            static constexpr int_t allocation_size = from_size * to_size + to_size;

            static inline void activate_host(FloatType * const & in, FloatType * const & out, FloatType * const & allocation) {
                for (int i = 0; i < to_size; i++) {
                    FloatType sum = 0;

                    for (int j = 0; j < from_size; j++) {
                        sum += in[j] * allocation[i * from_size + j];
                    }

                    out[i] = sum + allocation[from_size * to_size + i];
                }
            }
        };
    };
}