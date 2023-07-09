#pragma once

#include <cmath>

#include <pkai/activation_function/base.hpp>

namespace PKAI::ActivationFunction {
    struct Tanh : public ActivationFunction {
        template<typename FloatType>
        static inline FloatType call_host(FloatType x) {
            return std::tanh(x);
        }

        template<typename FloatType>
        __device__ static inline FloatType call_device(FloatType x) {
            return tanh(x);
        }
    };
}