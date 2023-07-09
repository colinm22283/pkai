#pragma once

#include <pkai/common.hpp>

namespace PKAI::Host {
    struct FullyConnected {
        static constexpr bool is_connection = true;

        template<typename Allocator, typename ActFunc, typename FloatType, nsize_t in_size, nsize_t out_size>
        struct Config {
            Allocator weights = Allocator(in_size * out_size);
            Allocator biases = Allocator(out_size);

            Allocator returned_costs = Allocator(in_size);

            inline Config() {
                for (int i = 0; i < in_size * out_size; i++) weights.data[i] = 0.25f;
            }

            inline void activate(FloatType * in, FloatType * out) {
                for (nsize_t i = 0; i < out_size; i++) {
                    out[i] = biases.data[i];

                    for (
                        FloatType * weight_ptr = weights.data + i * in_size, * in_ptr = in;
                        in_ptr < in + in_size;
                        in_ptr++, weight_ptr++
                    ) out[i] += *weight_ptr * *in_ptr;

                    out[i] = ActFunc::template call_host<FloatType>(out[i]);
                }
            }

            inline FloatType * backpropagate(FloatType * in, FloatType * out, FloatType * costs) {
                for (int i = 0; i < in_size; i++) returned_costs.data[i] = 0;

                for (int j = 0; j < out_size; j++) {
                    FloatType cost = costs[j];
                    FloatType & bias = biases.data[j];
                    FloatType * weight_ptr = weights.data + j * in_size;

                    for (int i = 0; i < in_size; i++) {
                        *weight_ptr += cost * in[i];

                        returned_costs.data[i] += cost * *weight_ptr;

                        weight_ptr++;
                    }

                    bias += cost;
                }

                return returned_costs.data;
            }
        };
    };
}
//
// L1
//    |/\|
// L2