#pragma once

#include <cmath>

#include <pkai/activation_function/base.hpp>

namespace PKAI::ActivationFunction {
    struct Sigmoid : public ActivationFunction {
        template<typename FloatType>
        static inline FloatType call_host(FloatType x) {
            return 1 / (1 + std::exp(-x));
        }

        template<typename FloatType>
        __device__ static inline FloatType call_device(FloatType x) {
            return 1 / (1 + exp(-x));
        }

        template<typename FloatType>
        static inline FloatType call_d_host(FloatType x) {
            FloatType e = std::exp(x);
            FloatType temp = (1 + e);
            return e / (temp * temp);
        }

        template<typename FloatType>
        __device__ static inline FloatType call_d_device(FloatType x) {
            FloatType e = exp(x);
            FloatType temp = (1 + e);
            return e / (temp * temp);
        }
    };
}