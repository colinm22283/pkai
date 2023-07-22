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

        template<typename FloatType>
        static inline FloatType call_d_host(FloatType x) {
            FloatType temp = 1 / std::cosh(x);
            return temp * temp;
        }

        template<typename FloatType>
        __device__ static inline FloatType call_d_device(FloatType x) {
            FloatType temp = 1 / cosh(x);
            return temp * temp;
        }
    };
}