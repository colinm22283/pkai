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
                std::uniform_real_distribution<FloatType> distro(0.25, 0.75);

                for (nsize_t i = 0; i < in_size * out_size; i++) weights.data[i] = distro(random_generator);
                for (nsize_t i = 0; i < out_size; i++) biases.data[i] = distro(random_generator);
            }

            inline void activate(FloatType * in_post_act, FloatType * out, FloatType * out_post_act) {
                FloatType * weight_ptr = weights.data;

                for (nsize_t i = 0; i < out_size; i++) {
                    out[i] = biases.data[i];

                    for (
                        FloatType * in_ptr = in_post_act;
                        in_ptr < in_post_act + in_size;
                        in_ptr++, weight_ptr++
                    ) out[i] += *weight_ptr * *in_ptr;

                    out[i] /= in_size;

                    out_post_act[i] = ActFunc::template call_host<FloatType>(out[i]);
                }
            }

            // costs are the cost derivatives with respect to outputs
            inline FloatType * backpropagate(FloatType * in, FloatType * in_post_act, FloatType * out, FloatType * out_post_act, FloatType * costs) {
//                std::cout << "\tOutputs: ";
//                for (int i = 0; i < out_size; i++) std::cout << out_post_act[i] << " ";
//                std::cout << "\n";
//                std::cout << "\tWeights: ";
//                for (int i = 0; i < in_size * out_size; i++) std::cout << weights.data[i] << " ";
//                std::cout << "\n";

                for (int i = 0; i < in_size; i++) returned_costs.data[i] = 0;

                FloatType * weight_ptr = weights.data;
                for (int j = 0; j < out_size; j++) {
                    FloatType & cost_d = costs[j];
                    FloatType & bias = biases.data[j];
                    FloatType act_d = ActFunc::template call_d_host<FloatType>(out[j]);

                    FloatType act_cost_d = cost_d * act_d;

                    for (int i = 0; i < in_size; i++) {
                        returned_costs.data[i] += act_cost_d * *weight_ptr;

                        *weight_ptr -= act_cost_d * in_post_act[i];

                        weight_ptr++;
                    }

                    bias -= act_cost_d;
                }

                return returned_costs.data;
            }

            inline void save(std::ofstream & fs) {
                fs.write((char *) weights.data, in_size * out_size * sizeof(FloatType));
                fs.write((char *) biases.data, out_size * sizeof(FloatType));
            }
            inline void load(std::ifstream & fs) {
                fs.read((char *) weights.data, in_size * out_size * sizeof(FloatType));
                fs.read((char *) biases.data, out_size * sizeof(FloatType));
            }
        };
    };
}
//
// L1
//    |/\|
// L2