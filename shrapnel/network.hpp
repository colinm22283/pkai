#pragma once

#include <pkai/common.hpp>
#include <pkai/layer.hpp>

namespace PKAI {
    template<typename VT, bool is_root, typename... Config> class _Network { };

    template<typename VT, typename InLayer, typename Connection, typename OutLayer, typename... Rest>
    class _Network<VT, true, InLayer, Connection, OutLayer, Rest...> {
    protected:
        InLayer::template Config<VT> in_layer;
        _Network<VT, false, OutLayer, Rest...> next;
        OutLayer::template Config<VT> & out_layer = next.in_layer;
        Connection::template Config<VT, InLayer::size, OutLayer::size> connection;

    public:
        inline void activate() {
            connection.activate(in_layer.neurons, out_layer.neurons);

            next.activate();
        }
        inline void backpropagate() {
            next.backpropagate();

            connection.backpropagate(in_layer.neurons, out_layer.neurons);
        }

        inline VT * in_neurons() noexcept { return in_layer.neurons; }
        inline VT * out_neurons() noexcept { return next.out_neurons(); }
    };
    template<typename VT, typename InLayer, typename Connection, typename OutLayer, typename... Rest>
    class _Network<VT, false, InLayer, Connection, OutLayer, Rest...> {
    public:
        InLayer::template Config<VT> in_layer;
        _Network<VT, false, OutLayer, Rest...> next;
        OutLayer::template Config<VT> & out_layer = next.in_layer;
        Connection::template Config<VT, InLayer::size, OutLayer::size> connection;

        inline void activate() {
            connection.activate(in_layer.neurons, out_layer.neurons);

            next.activate();
        }
        inline void backpropagate() {
            next.backpropagate();

            connection.backpropagate(in_layer.neurons, out_layer.neurons);
        }

        inline VT * out_neurons() noexcept { return next.out_neurons(); }
    };

    template<typename VT, typename InLayer, typename Connection, typename OutLayer>
    class _Network<VT, true, InLayer, Connection, OutLayer> {
    protected:
        InLayer::template Config<VT> in_layer;
        OutLayer::template Config<VT> out_layer;
        Connection::template Config<VT, InLayer::size, OutLayer::size> connection;

    public:
        inline void activate() {
            connection.activate(in_layer.neurons, out_layer.neurons);
        }
        inline void backpropagate() {
            connection.backpropagate(in_layer.neurons, out_layer.neurons);
        }

        inline VT * in_neurons() noexcept { return in_layer.neurons; }
        inline VT * out_neurons() noexcept { return out_layer.neurons; }
    };
    template<typename VT, typename InLayer, typename Connection, typename OutLayer>
    class _Network<VT, false, InLayer, Connection, OutLayer> {
    public:
        InLayer::template Config<VT> in_layer;
        OutLayer::template Config<VT> out_layer;
        Connection::template Config<VT, InLayer::size, OutLayer::size> connection;

        inline void activate() {
            connection.activate(in_layer.neurons, out_layer.neurons);
        }
        inline void backpropagate() {
            connection.backpropagate(in_layer.neurons, out_layer.neurons);
        }

        inline VT * out_neurons() noexcept { return out_layer.neurons; }
    };

    template<typename ValueType, typename... Config>
    using Network = _Network<ValueType, true, Config...>;
};