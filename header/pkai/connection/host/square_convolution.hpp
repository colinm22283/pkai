#pragma once

#include <pkai/common.hpp>

namespace PKAI::Host {
    template<nsize_t dimension, nsize_t width, nsize_t height, nsize_t channels>
    struct SquareConvolution {
        static constexpr bool is_connection = true;

        template<typename Allocator, typename ActFunc, typename FloatType, nsize_t in_size, nsize_t out_size>
        struct Config {
            static_assert(in_size >= dimension && in_size == width * height * channels && ((width - (dimension / 2)) * (height - (dimension / 2))) * channels == out_size, "Invalid convolution layer");

            static constexpr nsize_t out_width  = width - (dimension / 2);
            static constexpr nsize_t out_height = height - (dimension / 2);

            Allocator weights = Allocator(dimension * dimension * out_size);
            Allocator biases = Allocator(out_size);

            Allocator returned_costs = Allocator(in_size);

            inline Config() {
                std::uniform_real_distribution<FloatType> distro(-1, 1);

                for (nsize_t i = 0; i < dimension * dimension * out_size; i++) weights.data[i] = distro(random_generator);
                for (nsize_t i = 0; i < out_size; i++) biases.data[i] = distro(random_generator);
            }

            inline void activate(FloatType * in, FloatType * out) {
                FloatType * weight_ptr = weights.data;
                FloatType * in_ptr = in;

                for (nsize_t j = 0; j < channels; j++) {
                    for (nsize_t i = j * out_width * out_height; i < (j + 1) * out_width * out_height; i++) {
                        out[i] = biases.data[i];

                        for (nsize_t y = 0; y < dimension; y++) {
                            for (nsize_t x = 0; x < dimension; x++) {
                                out[i] += *weight_ptr * in_ptr[x + (y * width)];

                                weight_ptr++;
                            }
                        }

                        out[i] = ActFunc::template call_host<FloatType>(out[i] / (dimension * dimension));

                        in_ptr++;
                    }
                }
            }

            inline FloatType * backpropagate(FloatType * in, FloatType * out, FloatType * costs) { // costs are the cost derivatives with respect to outputs
                FloatType * weight_ptr = weights.data;
                FloatType * in_ptr = in;
                FloatType * rcost_ptr = returned_costs.data;

                for (nsize_t j = 0; j < channels; j++) {
                    for (nsize_t i = j * out_width * out_height; i < (j + 1) * out_width * out_height; i++) {
                        FloatType & cost = costs[i];
                        FloatType & bias = biases.data[i];

                        for (nsize_t y = 0; y < dimension; y++) {
                            for (nsize_t x = 0; x < dimension; x++) {
                                *weight_ptr += cost * in_ptr[x + (y * width)];

                                rcost_ptr[x + (y * width)] = cost * *weight_ptr * in_ptr[x + (y * width)];

                                weight_ptr++;
                            }
                        }

                        bias += cost;

                        in_ptr++;
                        rcost_ptr++;
                    }
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

//  1  2  3  4
//  5  6  7  9
// 10 11 12 13
// 14 15 16 17

// 0 1 2 2 3 4
