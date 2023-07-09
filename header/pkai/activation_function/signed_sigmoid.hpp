#pragma once

#include <cmath>

#include <pkai/activation_function/base.hpp>

namespace PKAI::ActivationFunction {
    struct SignedSigmoid : public ActivationFunction {
        template<typename FloatType>
        static inline FloatType call_host(FloatType x) {
            return (1 - std::exp(-x)) / (1 + std::exp(-x));
        }

        template<typename FloatType>
        __device__ static inline FloatType call_device(FloatType x) {
            return (1 - exp(-x)) / (1 + exp(-x));
        }
    };
}