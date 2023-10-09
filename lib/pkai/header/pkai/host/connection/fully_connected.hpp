#pragma once

#include <pkai/host/_host_check.hpp>

#include <pkai/universal/config.hpp>

namespace PKAI::Connection {
    template<typename _ActivationFunction>
    struct FullyConnected {
        static constexpr bool _is_connection_ = true;

        template<typename FloatType, int_t from_size, int_t to_size>
        struct Config {
            static constexpr int_t allocation_size = from_size * to_size + to_size;

            using ActivationFunction = _ActivationFunction::template Config<FloatType, from_size, to_size>;

            static inline void activate(FloatType * in, FloatType * out, FloatType * out_trans, FloatType * allocation) {
                for (int i = 0; i < to_size; i++) {
                    FloatType sum = 0;

                    for (int j = 0; j < from_size; j++) {
                        sum += in[j] * allocation[i * from_size + j];
                    }

                    out[i] = sum + allocation[from_size * to_size + i];
                    out_trans[i] = ActivationFunction::apply(out[i]);
                }
            }

            template<bool is_end>
            static inline void learn(
                FloatType * in,
                FloatType * out,
                FloatType * out_trans,
                FloatType * allocation,
                FloatType * cost_derivs, // dcost/dneuron_out
                FloatType * next_costs
            ) {
                if constexpr (!is_end) for (int j = 0; j < from_size; j++) next_costs[j] = 0;

                for (int i = 0; i < to_size; i++) {
                    FloatType trans_deriv = ActivationFunction::deriv_apply(out[i]); // dneuron_out/dneuron_in

                    FloatType move_factor = cost_derivs[i] * trans_deriv;

                    for (int j = 0; j < from_size; j++) {
                        FloatType weight_change = move_factor * in[j];

                        if constexpr (!is_end) next_costs[j] += weight_change * allocation[i * from_size + j];

                        allocation[i * from_size + j] -= 0.1 * weight_change;
                    }

                    allocation[from_size * to_size + i] -= 0.1 * move_factor;
                }
            }
        };
    };
}