#pragma once

#include <pkai/universal/config.hpp>

namespace PKAI::ActivationFunction {
    struct LeakyReLu {
        template<typename FloatType, int_t from_size, int_t to_size>
        struct Config {
#ifdef PKAI_HOST
            static constexpr inline FloatType apply(FloatType x) {
                if (x > 0) return x;
                else return 0.1 * x;
            }
            static constexpr inline FloatType deriv_apply(FloatType x) {
                if (x > 0) return 1;
                else return 0.1;
            }
#endif
#ifdef PKAI_DEVICE
            __device__ static constexpr inline FloatType apply(FloatType x) {
                if (x > 0) return x;
                else return 0.1 * x;
            }
            __device__ static constexpr inline FloatType deriv_apply(FloatType x) {
                if (x > 0) return 1;
                else return 0.1;
            }
#endif
        };
    };
}