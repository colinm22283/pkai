#pragma once

#include "config.hpp"

#ifndef __device__
    #define __device__
#endif
#ifndef __host__
    #define __host__
#endif

namespace PKAI {
    template<typename Allocator, typename FloatType, typename... Info>
    class Network {
    protected:
        template<typename IL, typename C, typename OL, typename... Rest>
        struct layer_container_t {
            Allocator::template Allocate<FloatType, IL::size> layer_in_allocation;

            layer_container_t<OL, Rest...> next = layer_container_t<OL, Rest...>(layer_in_allocation.data());

            static constexpr int_t in_size = IL::size;
            [[nodiscard]] FloatType * const & in_neurons() const noexcept { return layer_in_allocation.data(); }
            static constexpr int_t out_size = OL::size;
            [[nodiscard]] FloatType * const & out_neurons() const noexcept { return next.in_neurons(); }

            inline auto & last_layer() { return next.last_layer(); }

            using Connection = C::template Config<FloatType, in_size, out_size>;
            Allocator::template Allocate<FloatType, Connection::allocation_size> connection_allocation;

            inline void activate_host() {
                static_assert(!C::_is_host, "Connection must be host to host activate");
                Connection::activate_host(in_neurons(), out_neurons(), connection_allocation.data());
                next.activate_hosy();
            }
        };
        template<typename IL, typename C, typename OL>
        struct layer_container_t<IL, C, OL> {
            Allocator::template Allocate<FloatType, IL::size> layer_in_allocation;
            Allocator::template Allocate<FloatType, IL::size> layer_out_allocation;

            static constexpr int_t in_size = IL::size;
            [[nodiscard]] FloatType * const & in_neurons() const noexcept { return layer_in_allocation.data(); }
            [[nodiscard]] __device__ float * const & in_neurons_dev() const noexcept { return layer_in_allocation.data(); }
            static constexpr int_t out_size = OL::size;
            [[nodiscard]] FloatType * const & out_neurons() const noexcept { return layer_out_allocation.data(); }

            inline auto & last_layer() { return *this; }

            using Connection = C::template Config<FloatType, in_size, out_size>;
            Allocator::template Allocate<FloatType, Connection::allocation_size> connection_allocation;

            inline void activate_host() {
                static_assert(!C::_is_host, "Connection must be host to host activate");
                Connection::activate_host(in_neurons(), out_neurons(), connection_allocation.data());
            }
        };

        layer_container_t<Info...> layers;

    public:
        inline void set_inputs(FloatType const * inputs) {
            layers.layer_in_allocation.set_data(inputs, layers.in_size);
        }
        inline void activate_host() {
            layers.activate_host();
        }
        inline void get_outputs(FloatType * outputs) {
            layers.last_layer().layer_out_allocation.get_data(outputs, layers.last_layer().out_size);
        }
    };
}

