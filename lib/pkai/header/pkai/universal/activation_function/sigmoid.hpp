#pragma once

#include <cmath>

#include <pkai/universal/config.hpp>

namespace PKAI::ActivationFunction {
    struct Sigmoid {
        template<typename FloatType, int_t from_size, int_t to_size>
        struct Config {
#ifdef PKAI_HOST
            static constexpr inline FloatType apply(FloatType x) {
                return 1 / (1 + std::exp(-x));
            }
            static constexpr inline FloatType deriv_apply(FloatType x) {
                FloatType ex = std::exp(-x);
                return ex / ((ex * ex) + 2 * ex + 1);
            }
#endif
#ifdef PKAI_DEVICE
            __device__ static constexpr inline FloatType apply(FloatType x) {
                return 1 / (1 + exp(-x));
            }
            __device__ static constexpr inline FloatType deriv_apply(FloatType x) {
                FloatType ex = exp(-x);
                return ex / ((ex * ex) + 2 * ex + 1);
            }
#endif
        };
    };
}