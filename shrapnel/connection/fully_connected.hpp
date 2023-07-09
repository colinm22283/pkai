#pragma once

#include <pkai/common.hpp>

namespace PKAI {
    namespace _Kernels {
        template<typename VT, lsize_t in_size, lsize_t out_size>
        __global__
        void fully_connected_activate_k(VT * in, VT * out, VT * biases, VT * weights) {
            lsize_t out_index = blockDim.x * blockIdx.x + threadIdx.x;

            out[out_index] = biases[out_index];

            for (lsize_t i = 0; i < in_size; i++) {
                out[out_index] += in[i] * weights[i + out_index * out_size];
            }
        }
    }

    struct FullyConnected {
        template<typename VT, lsize_t in_size, lsize_t out_size>
        struct Config {
            VT * biases;
            VT * weights;

            Config():
                weights(dev::alloc_array<VT>(in_size * out_size)),
                biases(dev::alloc_array<VT>(out_size)) { }
            ~Config() {
                dev::free_array(weights, in_size * out_size);
                dev::free_array(biases, out_size);
            }

            inline void activate(VT * in, VT * out) {
                _Kernels::fully_connected_activate_k<VT, in_size, out_size><<<1, out_size>>>(in, out, biases, weights);
            }
            inline void activate(cudaStream_t stream, VT * in, VT * out) {
                _Kernels::fully_connected_activate_k<VT, in_size, out_size><<<1, out_size, 0, stream>>>(in, out, biases, weights);
            }

            inline void backpropagate(VT * in, VT * out) {

            }

            inline void randomize_weights() {
                // TODO
            }
        };
    };
}