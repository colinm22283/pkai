#pragma once

#include <pkai/universal/config.hpp>

namespace PKAI::ActivationFunction {
    template<int_t step_pos_percent>
    struct Step {
        template<typename FloatType, int_t from_size, int_t to_size>
        struct Config {
            static constexpr FloatType step_pos = (FloatType) step_pos_percent / 100.0;

#ifdef PKAI_HOST
            static constexpr inline FloatType apply(FloatType x) {
                if (x >= step_pos) return 1;
                else return 0;
            }
            static constexpr inline FloatType deriv_apply(FloatType x) {
                if (x == step_pos) return 1;
                else return 0;
            }
#endif
#ifdef PKAI_DEVICE
            __device__ static constexpr inline FloatType apply(FloatType x) {
                if (x >= step_pos) return 1;
                else return 0;
            }
            __device__ static constexpr inline FloatType deriv_apply(FloatType x) {
                if (x == step_pos) return 1;
                else return 0;
            }
#endif
        };
    };
}