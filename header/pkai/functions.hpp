#pragma once

#include <cmath>

#include <cuda_runtime_api.h>

namespace PKAI {
    __device__
    inline float sigmoid(float x) {
        return 1 / (1 + powf(M_E, -x));
    }
    __device__
    inline float sigmoid_prime(float x) {
        return x * (1.0f - x);
    }
    __device__
    inline float arc_sigmoid(float x) {
        return -logf(1 / x - 1);
    }

    __device__
    inline float fast_sigmoid(float x) {
        return x / (1 + fabs(x));
    }
    __device__
    inline float fast_sigmoid_prime(float x) {
        return -1 / (x * x + 2 * fabs(x) + 1);
    }

    __device__
    inline float tanh(float x) {
        return tanhf(x);
    }
    __device__
    inline float arc_tanh(float x) {
        return atanhf(x);
    }

    __device__
    inline float cube(float x) {
        return x * x * x;
    }

    __device__
    inline float relu(float x) {
        return x > 0 ? x : 0;
    }

    __device__
    inline float crelu(float x) {
        if (x > 0) {
            return x < 1 ? x : 1;
        } else return 0;
    }
}