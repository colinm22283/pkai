#pragma once

#include <cmath>

#include <pkai/activation_function/base.hpp>

namespace PKAI::ActivationFunction {
    struct SignedFastSigmoid : public ActivationFunction {
        template<typename FloatType>
        static inline FloatType call_host(FloatType x) {
            return x / (1 + std::abs(x));
        }

        template<typename FloatType>
        __device__ static inline FloatType call_device(FloatType x) {
            return x / (1 + abs(x));
        }
    };
}