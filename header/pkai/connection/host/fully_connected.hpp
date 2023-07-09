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
                std::uniform_real_distribution<FloatType> distro(-1, 1);

                for (nsize_t i = 0; i < in_size * out_size; i++) weights.data[i] = distro(random_generator);
                for (nsize_t i = 0; i < out_size; i++) biases.data[i] = distro(random_generator);
            }

            inline void activate(FloatType * in, FloatType * out) {
                FloatType * weight_ptr = weights.data;

                for (nsize_t i = 0; i < out_size; i++) {
                    out[i] = biases.data[i];

                    for (
                        FloatType * in_ptr = in;
                        in_ptr < in + in_size;
                        in_ptr++, weight_ptr++
                    ) out[i] += *weight_ptr * *in_ptr;

                    out[i] = ActFunc::template call_host<FloatType>(out[i] / in_size);
                }
            }

            inline FloatType * backpropagate(FloatType * in, FloatType * out, FloatType * costs) {
//                std::cout << "Weights: ";
//                for (int i = 0; i < in_size * out_size; i++) std::cout << weights.data[i] << " ";
//                std::cout << "\n";
//                std::cout << "Inputs: ";
//                for (int i = 0; i < in_size; i++) std::cout << in[i] << " ";
//                std::cout << "\n\n";

                for (int i = 0; i < in_size; i++) returned_costs.data[i] = 0;

                FloatType * weight_ptr = weights.data;
                for (int j = 0; j < out_size; j++) {
                    FloatType cost = costs[j];
                    FloatType & bias = biases.data[j];

                    for (int i = 0; i < in_size; i++) {
                        *weight_ptr += 0.1 * cost * in[i];

                        returned_costs.data[i] += cost * *weight_ptr;

                        weight_ptr++;
                    }

                    bias += 0.1 * cost;
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