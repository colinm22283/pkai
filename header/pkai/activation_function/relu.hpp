#pragma once

#include <cmath>

#include <pkai/activation_function/base.hpp>

namespace PKAI::ActivationFunction {
    struct ReLu : public ActivationFunction {
        template<typename FloatType>
        static inline FloatType call_host(FloatType x) {
            return std::max(0.0f, x);
        }

        template<typename FloatType>
        __device__ static inline FloatType call_device(FloatType x) {
            return max(0.0f, x);
        }
    };
}