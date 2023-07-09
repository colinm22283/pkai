#pragma once

namespace PKAI::ActivationFunction {
    struct ActivationFunction {
        static constexpr bool is_activation_function = true;

        template<typename FloatType>
        static inline FloatType call_host(FloatType x) { return 0; }

        template<typename FloatType>
        __device__ static inline FloatType call_device(FloatType x) { return 0; }
    };
}