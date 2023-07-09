#pragma once

#include <cmath>

#include <pkai/activation_function/base.hpp>

namespace PKAI::ActivationFunction {
    struct FastSigmoid : public ActivationFunction {
        template<typename FloatType>
        static inline FloatType call_host(FloatType x) {
            return 0.5f * (x / (1 + std::abs(x)) + 1);
        }

        template<typename FloatType>
        __device__ static inline FloatType call_device(FloatType x) {
            return 0.5f * (x / (1 + abs(x)) + 1);
        }
    };
}