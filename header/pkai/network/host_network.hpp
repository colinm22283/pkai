#pragma once

#include <cstring>

#include <pkai/network/network.hpp>
#include <pkai/common.hpp>
#include <pkai/allocators.hpp>

#include <device/heap.hpp>

namespace PKAI {
    template<typename ActivationFunction, typename FloatType, typename... Config>
    class HostNetwork : public _Network<_host_allocator_t<FloatType>, ActivationFunction, FloatType, true, Config...> {
        using base_t = _Network<_host_allocator_t<FloatType>, ActivationFunction, FloatType, true, Config...>;

    public:
        using base_t::input_size;
        using base_t::backpropagate_recur; // TODO This too
        using base_t::output_ptr; // TODO: move to protected

        inline void give_input(FloatType * data) {
            std::memcpy(
                this->in_layer.data,
                data,
                base_t::static_input_size * sizeof(FloatType)
            );
        }
        inline void copy_output(FloatType * data) {
            std::memcpy(
                data,
                this->output_ptr(),
                base_t::static_output_size * sizeof(FloatType)
            );
        }

        inline void backpropagate(FloatType * correct) {
            FloatType costs[base_t::static_input_size];
            for (nsize_t i = 0; i < base_t::static_input_size; i++) {
                FloatType temp = correct[i] - output_ptr()[i];
                costs[i] = temp;
            }

            backpropagate_recur(costs);
        }
    };
}