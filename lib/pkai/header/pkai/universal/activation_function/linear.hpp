#pragma once

#include <pkai/universal/config.hpp>

namespace PKAI::ActivationFunction {
    struct Linear {
        template<typename FloatType, int_t from_size, int_t to_size>
        struct Config {
#ifdef PKAI_HOST
            static constexpr inline FloatType apply(FloatType x) { return x; }
            static constexpr inline FloatType deriv_apply(FloatType x) { return 1; }
#endif
#ifdef PKAI_DEVICE
            __device__ static constexpr inline FloatType apply(FloatType x) { return x; }
            __device__ static constexpr inline FloatType deriv_apply(FloatType x) { return 1; }
#endif
        };
    };
}