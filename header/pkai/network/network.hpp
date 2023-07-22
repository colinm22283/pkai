#pragma once

#include <fstream>

#include <pkai/layer.hpp>

namespace PKAI {
    // ArrayAllocator must contain a member .data with is of type "FloatType *"
    template<typename Allocator, typename ActFunc, typename FloatType, bool is_head, typename... Config> class _Network;

    template<typename Allocator, typename ActFunc, typename FloatType, typename InLayer, typename Connection, typename OutLayer, typename... Rest>
    class _Network<Allocator, ActFunc, FloatType, true, InLayer, Connection, OutLayer, Rest...> {
        static_assert(InLayer::is_layer && Connection::is_connection && OutLayer::is_layer, "Invalid config types!");
        static_assert(ActFunc::is_activation_function, "Invalid activation function type");

    protected:
        using next_t = _Network<Allocator, ActFunc, FloatType, false, OutLayer, Rest...>;

        Allocator in_layer = Allocator(InLayer::size);
        Connection::template Config<Allocator, ActFunc, FloatType, InLayer::size, OutLayer::size> connection;
        next_t next;
        Allocator & out_layer = next.in_layer;
        Allocator & out_layer_post_act = next.in_layer_post_act;

        static constexpr nsize_t static_input_size = InLayer::size;
        static constexpr nsize_t static_output_size = next_t::static_output_size;

        inline FloatType * output_ptr() { return next.output_ptr(); }

        // backpropagate this layer and the next and returns the cost
        inline FloatType * backpropagate_recur(FloatType * costs) {
            return connection.backpropagate(in_layer.data, in_layer.data, out_layer.data, out_layer_post_act.data, next.backpropagate_recur(costs));
        }

        inline void save_recur(std::ofstream & fs) {
            connection.save(fs);
            next.save_recur(fs);
        }
        inline void load_recur(std::ifstream & fs) {
            connection.load(fs);
            next.load_recur(fs);
        }

    public:
        [[nodiscard]] consteval nsize_t input_size() const noexcept { return InLayer::size; }
        [[nodiscard]] consteval nsize_t output_size() const noexcept { return static_output_size; }

        inline void activate() {
            connection.activate(in_layer.data, out_layer.data, out_layer_post_act.data);
            next.activate();
        }

        inline void save(const char * path) {
            std::ofstream fs(path, std::ofstream::trunc);

            save_recur(fs);
        }
        inline void load(const char * path) {
            std::ifstream fs(path);
            load_recur(fs);
        }
    };

    template<typename Allocator, typename ActFunc, typename FloatType, typename InLayer, typename Connection, typename OutLayer>
    class _Network<Allocator, ActFunc, FloatType, true, InLayer, Connection, OutLayer> {
        static_assert(InLayer::is_layer && Connection::is_connection && OutLayer::is_layer, "Invalid config types!");
        static_assert(ActFunc::is_activation_function, "Invalid activation function type");

    protected:
        Allocator in_layer = Allocator(InLayer::size);
        Connection::template Config<Allocator, ActFunc, FloatType, InLayer::size, OutLayer::size> connection;
        Allocator out_layer = Allocator(OutLayer::size);
        Allocator out_layer_post_act = Allocator(OutLayer::size);

        static constexpr nsize_t static_input_size = InLayer::size;
        static constexpr nsize_t static_output_size = OutLayer::size;

        inline FloatType * output_ptr() { return out_layer_post_act.data; }

        // backpropagate this layer and the next and returns the cost
        inline FloatType * backpropagate_recur(FloatType * costs) {
            return connection.backpropagate(in_layer.data, in_layer.data, out_layer.data, out_layer_post_act.data, costs);
        }

        inline void save_recur(std::ofstream & fs) {
            connection.save(fs);
        }
        inline void load_recur(std::ifstream & fs) {
            connection.load(fs);
        }

    public:
        [[nodiscard]] consteval nsize_t input_size() const noexcept { return InLayer::size; }
        [[nodiscard]] consteval nsize_t output_size() const noexcept { return static_output_size; }

        inline void activate() {
            connection.activate(in_layer.data, out_layer.data, out_layer_post_act.data);
        }

        inline void save(const char * path) {
            std::ofstream fs(path, std::ofstream::trunc);

            save_recur(fs);
        }
    };

    template<typename Allocator, typename ActFunc, typename FloatType, typename InLayer, typename Connection, typename OutLayer, typename... Rest>
    class _Network<Allocator, ActFunc, FloatType, false, InLayer, Connection, OutLayer, Rest...> {
    public:
        using next_t = _Network<Allocator, ActFunc, FloatType, false, OutLayer, Rest...>;

        Allocator in_layer = Allocator(InLayer::size);
        Allocator in_layer_post_act = Allocator(InLayer::size);
        Connection::template Config<Allocator, ActFunc, FloatType, InLayer::size, OutLayer::size> connection;
        next_t next;
        Allocator & out_layer = next.in_layer;
        Allocator & out_layer_post_act = next.in_layer_post_act;

        static constexpr nsize_t static_input_size = InLayer::size;
        static constexpr nsize_t static_output_size = next_t::static_output_size;

        inline FloatType * output_ptr() { return next.output_ptr(); }

        // backpropagate this layer and the next and returns the cost
        inline FloatType * backpropagate_recur(FloatType * costs) {
            return connection.backpropagate(in_layer.data, in_layer_post_act.data, out_layer.data, out_layer_post_act.data, next.backpropagate_recur(costs));
        }

        inline void activate() {
            connection.activate(in_layer_post_act.data, out_layer.data, out_layer_post_act.data);
            next.activate();
        }

        inline void save_recur(std::ofstream & fs) {
            connection.save(fs);
            next.save_recur(fs);
        }
        inline void load_recur(std::ifstream & fs) {
            connection.load(fs);
            next.load_recur(fs);
        }
    };
    template<typename Allocator, typename ActFunc, typename FloatType, typename InLayer, typename Connection, typename OutLayer>
    class _Network<Allocator, ActFunc, FloatType, false, InLayer, Connection, OutLayer> {
    public:
        Allocator in_layer = Allocator(InLayer::size);
        Allocator in_layer_post_act = Allocator(InLayer::size);
        Connection::template Config<Allocator, ActFunc, FloatType, InLayer::size, OutLayer::size> connection;
        Allocator out_layer = Allocator(OutLayer::size);
        Allocator out_layer_post_act = Allocator(OutLayer::size);

        static constexpr nsize_t static_input_size = InLayer::size;
        static constexpr nsize_t static_output_size = OutLayer::size;

        inline FloatType * output_ptr() { return out_layer_post_act.data; }

        // backpropagate this layer and the next and returns the cost
        inline FloatType * backpropagate_recur(FloatType * costs) {
            return connection.backpropagate(in_layer.data, in_layer_post_act.data, out_layer.data, out_layer_post_act.data, costs);
        }

        inline void activate() {
            connection.activate(in_layer_post_act.data, out_layer.data, out_layer_post_act.data);
        }

        inline void save_recur(std::ofstream & fs) {
            connection.save(fs);
        }
        inline void load_recur(std::ifstream & fs) {
            connection.load(fs);
        }
    };
}
