#pragma once

#include <pkai/universal/config.hpp>
#include <pkai/universal/dataset.hpp>

#ifdef PKAI_DEVICE
#include <cuda_runtime.h>
#endif

namespace PKAI {
    template<typename Allocator, typename FloatType, typename... Info>
    class Network {
    protected:
        template<typename prev_t, typename IL, typename C, typename OL, typename... Rest>
        struct layer_container_t {
            Allocator::template Instance<FloatType, IL::size> layer_in_allocation;
            Allocator::template Instance<FloatType, IL::size> layer_out_allocation;

            layer_container_t<
                layer_container_t<prev_t, IL, C, OL, Rest...>,
                OL,
                Rest...
            > next = layer_container_t<
                layer_container_t<prev_t, IL, C, OL, Rest...>,
                OL,
                Rest...
            >(*this);
            prev_t & prev;
            explicit inline layer_container_t(prev_t & _prev): prev(_prev) { }

            static constexpr int_t in_size = IL::size;
            [[nodiscard]] FloatType * in_neurons() noexcept { return layer_in_allocation.data(); }
            static constexpr int_t out_size = OL::size;
            [[nodiscard]] FloatType * out_neurons() noexcept { return layer_out_allocation.data(); }
            [[nodiscard]] FloatType * out_neurons_trans() noexcept { return next.in_neurons(); }

            inline auto & last_layer() { return next.last_layer(); }

            using Connection = C::template Config<FloatType, in_size, out_size>;
            Allocator::template Instance<FloatType, Connection::allocation_size> connection_allocation;

#ifdef PKAI_HOST
            inline void activate() {
                Connection::activate(in_neurons(), out_neurons(), out_neurons_trans(), connection_allocation.data());
                next.activate();
            }
            inline void learn(FloatType * costs) {
                FloatType temp[in_size];
                Connection::template learn<false>(in_neurons(), out_neurons(), out_neurons_trans(), connection_allocation.data(), costs, temp);
                prev.learn(temp);
            }
#endif
#ifdef PKAI_DEVICE
            __device__
            inline void activate(int idx) {
                Connection::activate(in_neurons(), out_neurons(), connection_allocation.data());
            }
            __device__
            inline void learn(int idx) {
                Connection::activate(in_neurons(), out_neurons(), connection_allocation.data());
            }
#endif
        };
        template<typename prev_t, typename IL, typename C, typename OL>
        struct layer_container_t<prev_t, IL, C, OL> {
            Allocator::template Instance<FloatType, IL::size> layer_in_allocation;
            Allocator::template Instance<FloatType, IL::size> layer_out_allocation;
            Allocator::template Instance<FloatType, IL::size> layer_out_trans_allocation;

            prev_t & prev;
            explicit inline layer_container_t(prev_t & _prev): prev(_prev) { }

            static constexpr int_t in_size = IL::size;
            [[nodiscard]] FloatType * in_neurons() noexcept { return layer_in_allocation.data(); }
            static constexpr int_t out_size = OL::size;
            [[nodiscard]] FloatType * out_neurons() noexcept { return layer_out_allocation.data(); }
            [[nodiscard]] FloatType * out_neurons_trans() noexcept { return layer_out_trans_allocation.data(); }

            inline auto & last_layer() { return *this; }

            using Connection = C::template Config<FloatType, in_size, out_size>;
            Allocator::template Instance<FloatType, Connection::allocation_size> connection_allocation;

#ifdef PKAI_HOST
            inline void activate() {
                Connection::activate(in_neurons(), out_neurons(), out_neurons_trans(), connection_allocation.data());
            }
            inline void learn(FloatType * costs) {
                FloatType temp[in_size];
                Connection::template learn<false>(in_neurons(), out_neurons(), out_neurons_trans(), connection_allocation.data(), costs, temp);
                prev.learn(temp);
            }
#endif
#ifdef PKAI_DEVICE
            __device__
            inline void activate(int idx) {
                Connection::activate(in_neurons(), out_neurons(), connection_allocation.data());
            }
            __device__
            inline void learn(int idx) {
                Connection::activate(in_neurons(), out_neurons(), connection_allocation.data());
            }
#endif
        };
        template<typename IL, typename C, typename OL, typename... Rest>
        struct layer_container_base_t {
            Allocator::template Instance<FloatType, IL::size> layer_in_allocation;
            Allocator::template Instance<FloatType, OL::size> layer_out_allocation;

            layer_container_t<
                layer_container_base_t<IL, C, OL, Rest...>,
                OL,
                Rest...
            > next = layer_container_t<
                layer_container_base_t<IL, C, OL, Rest...>,
                OL,
                Rest...
            >(*this);

            static constexpr int_t in_size = IL::size;
            [[nodiscard]] FloatType * in_neurons() noexcept { return layer_in_allocation.data(); }
            static constexpr int_t out_size = OL::size;
            [[nodiscard]] FloatType * out_neurons() noexcept { return layer_out_allocation.data(); }
            [[nodiscard]] FloatType * out_neurons_trans() noexcept { return next.in_neurons(); }

            inline auto & last_layer() { return next.last_layer(); }

            using Connection = C::template Config<FloatType, in_size, out_size>;
            Allocator::template Instance<FloatType, Connection::allocation_size> connection_allocation;

#ifdef PKAI_HOST
            inline void activate() {
                Connection::activate(in_neurons(), out_neurons(), out_neurons_trans(), connection_allocation.data());
                next.activate();
            }
            inline void learn(FloatType * costs) {
                Connection::template learn<true>(in_neurons(), out_neurons(), out_neurons_trans(), connection_allocation.data(), costs, nullptr);
            }
#endif
#ifdef PKAI_DEVICE
            __device__
            inline void activate(int idx) {
                Connection::activate(in_neurons(), out_neurons(), connection_allocation.data());
            }
            __device__
            inline void learn(int idx) {
                Connection::activate(in_neurons(), out_neurons(), connection_allocation.data());
            }
#endif
        };
        template<typename IL, typename C, typename OL>
        struct layer_container_base_t<IL, C, OL> {
            Allocator::template Instance<FloatType, IL::size> layer_in_allocation;
            Allocator::template Instance<FloatType, IL::size> layer_out_allocation;
            Allocator::template Instance<FloatType, IL::size> layer_out_trans_allocation;

            static constexpr int_t in_size = IL::size;
            [[nodiscard]] FloatType * in_neurons() noexcept { return layer_in_allocation.data(); }
            static constexpr int_t out_size = OL::size;
            [[nodiscard]] FloatType * out_neurons() noexcept { return layer_out_allocation.data(); }
            [[nodiscard]] FloatType * out_neurons_trans() noexcept { return layer_out_trans_allocation.data(); }

            inline auto & last_layer() { return *this; }

            using Connection = C::template Config<FloatType, in_size, out_size>;
            Allocator::template Instance<FloatType, Connection::allocation_size> connection_allocation;

#ifdef PKAI_HOST
            inline void activate() {
                Connection::activate(in_neurons(), out_neurons(), out_neurons_trans(), connection_allocation.data());
            }
            inline void learn(FloatType * costs) {
                Connection::template learn<false>(in_neurons(), out_neurons(), out_neurons_trans(), connection_allocation.data(), costs, nullptr);
            }
#endif
#ifdef PKAI_DEVICE
            __device__
            inline void activate(int idx) {
                Connection::activate(in_neurons(), out_neurons(), connection_allocation.data());
            }
            __device__
            inline void learn(int idx) {
                Connection::activate(in_neurons(), out_neurons(), connection_allocation.data());
            }
#endif
        };

        layer_container_base_t<Info...> layers;

    public:
        inline void set_inputs(FloatType const * inputs) {
            layers.layer_in_allocation.set_data(inputs, layers.in_size);
        }
        inline void get_outputs(FloatType * outputs) {
            layers.last_layer().layer_out_trans_allocation.get_data(outputs, layers.last_layer().out_size);
        }

#ifdef PKAI_HOST
        inline void activate() {
            layers.activate();
        }
        inline void learn(FloatType const * correct) {
            FloatType cost_derivs[layers.last_layer().out_size];

            for (int i = 0; i < layers.last_layer().out_size; i++) {
                cost_derivs[i] = 2 * (layers.last_layer().out_neurons_trans()[i] - correct[i]);
            }

            layers.last_layer().learn(cost_derivs);
        }
        template<int progress_report_interval = 0>
        inline void train(auto & dataset, int_t iterations) {
            for (int i = 0; i < iterations; i++) {
                if constexpr (progress_report_interval != 0) if (iterations % progress_report_interval == 0) {
                    std::cout << i << "/" << iterations << ": " << total_cost(dataset) << "\n";
                }

                auto & set = dataset[i % dataset.size()];

                set_inputs(set.in());
                activate();
                learn(set.out());
            }
        }

        inline FloatType cost(FloatType const * correct) {
            FloatType cost = 0;

            for (int i = 0; i < layers.last_layer().out_size; i++) {
                FloatType temp = correct[i] - layers.last_layer().out_neurons_trans()[i];
                cost += temp * temp;
            }

            return cost / layers.last_layer().out_size;
        }
        inline FloatType total_cost(auto & dataset) {
            FloatType cost_sum = 0;

            for (int i = 0; i < dataset.size(); i++) {
                set_inputs(dataset[i].in());
                activate();
                cost_sum += cost(dataset[i].out());
            }

            return cost_sum / dataset.size();
        }
#endif
#ifdef PKAI_DEVICE
    protected:
        __global__
        template<typename T>
        static inline void activate_k() {

        }

    public:
        inline void activate() {

        }
        inline void train() {

        }
#endif
    };
}

