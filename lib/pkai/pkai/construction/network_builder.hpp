#pragma once

#include "../network.hpp"

#include "layer.hpp"

namespace PKAI {
    namespace _NetworkBuilderPrivate {
        struct network_builder_state_t {
            bool has_allocator = false;
            bool has_float_type = false;
            bool layers_satisfied = false;
            int layer_count = 0;
        };
        struct NoneType { };
    }

    template<_NetworkBuilderPrivate::network_builder_state_t state, typename Allocator, typename FloatType, typename... Data>
    struct _NetworkBuilder {
    protected:
        template<_NetworkBuilderPrivate::network_builder_state_t _state, typename... _Data>
        struct builder_t {
            using Build = Network<_Data...>;
        };

        template<_NetworkBuilderPrivate::network_builder_state_t _state, typename... _Data>
        static consteval builder_t<_state, _Data...> build() {
            static_assert(_state.has_allocator, "Network must have an allocator!");
            static_assert(_state.has_float_type, "Network must have a float type!");
            static_assert(_state.layers_satisfied, "All layers must be satisfied!");

            static_assert(
                Allocator::_is_host && (Data::_is_host && ...) ||
                Allocator::_is_device && (Data::_is_device && ...),
                "All elements must be have either host or device support"
            );

            return {};
        }

    public:
        using Build = decltype(build<state, Allocator, FloatType, Data...>())::Build;

        template<typename _Allocator> requires(_Allocator::_is_allocator)
        using DefineAllocator = _NetworkBuilder<_NetworkBuilderPrivate::network_builder_state_t {
            .has_allocator = true,
            .has_float_type = state.has_float_type,
            .layers_satisfied = state.layers_satisfied,
            .layer_count = state.layer_count,
        }, _Allocator, FloatType, Data...>;

        template<typename _FloatType>
        using DefineFloatType = _NetworkBuilder<_NetworkBuilderPrivate::network_builder_state_t {
            .has_allocator = state.has_allocator,
            .has_float_type = true,
            .layers_satisfied = state.layers_satisfied,
            .layer_count = state.layer_count + 1,
        }, Allocator, _FloatType, Data...>;

        template<int_t size>
        using AddLayer = _NetworkBuilder<_NetworkBuilderPrivate::network_builder_state_t {
            .has_allocator = state.has_allocator,
            .has_float_type = state.has_float_type,
            .layers_satisfied = (state.layer_count > 0),
            .layer_count = state.layer_count + 1,
        }, Allocator, FloatType, Data..., Layer<size>>;

        template<typename Connection>
        using AddConnection = _NetworkBuilder<_NetworkBuilderPrivate::network_builder_state_t {
            .has_allocator = state.has_allocator,
            .has_float_type = state.has_float_type,
            .layers_satisfied = false,
            .layer_count = state.layer_count,
        }, Allocator, FloatType, Data..., Connection>;
    };

    using NetworkBuilder = _NetworkBuilder<
        _NetworkBuilderPrivate::network_builder_state_t { },
        _NetworkBuilderPrivate::NoneType,
        _NetworkBuilderPrivate::NoneType
    >;
}