#pragma once

#include "config.hpp"

namespace PKAI {
    template<typename Allocator, typename FloatType, typename... Info>
    class Network {
    protected:
        template<typename IL, typename C, typename OL, typename... Rest>
        struct layer_container_t {
            Allocator::template Allocate<FloatType, IL::size> layer_in_allocation;

            static constexpr int_t in_size = IL::size;
            float * const & in_neurons = layer_in_allocation.data();

            layer_container_t<OL, Rest...> next = layer_container_t<OL, Rest...>(layer_in_allocation.data());

            static constexpr int_t out_size = OL::size;
            float * const & out_neurons = next.layer_in_allocation.data();

            inline auto & last_layer() { return next.last_layer(); }
        };
        template<typename IL, typename C, typename OL>
        struct layer_container_t<IL, C, OL> {
            Allocator::template Allocate<FloatType, IL::size> layer_in_allocation;
            Allocator::template Allocate<FloatType, IL::size> layer_out_allocation;

            static constexpr int_t in_size = IL::size;
            float * const & in_neurons = layer_in_allocation.data();
            static constexpr int_t out_size = OL::size;
            float * const & out_neurons = layer_out_allocation.data();

            inline auto & last_layer() { return *this; }
        };

        layer_container_t<Info...> layers;

    public:
        inline void set_inputs(FloatType const * inputs) {
            layers.layer_in_allocation.set_data(inputs, layers.in_size);
        }
        inline void activate() {

        }
        inline void get_outputs(FloatType * outputs) {
            layers.last_layer().layer_out_allocation.get_data(outputs, layers.last_layer().out_size);
        }
    };
}