#pragma once

#include <cuda_runtime.h>

#include <pkai/device/_device_check.hpp>

#include <pkai/universal/config.hpp>

namespace PKAI::Connection {
    struct FullyConnected {
        static constexpr bool _is_connection_ = true;

        template<typename FloatType, int_t from_size, int_t to_size>
        struct Config {
            static constexpr int_t allocation_size = from_size * to_size + to_size;

            __device__ static inline void activate(FloatType * in, FloatType * out, FloatType * allocation) {
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