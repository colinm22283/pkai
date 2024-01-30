#pragma once

#ifndef PKAI_HOST
#ifndef PKAI_DEVICE
#error PKAI_HOST or PKAI_DEVICE must be defined
#endif
#endif

#include <pkai/universal/network.hpp>
#include <pkai/universal/layer.hpp>

#ifdef PKAI_HOST
#include <pkai/host/allocator.hpp>
#endif
#ifdef PKAI_DEVICE
#include <pkai/device/allocator.hpp>
#endif

namespace PKAI {
    namespace _NetworkBuilderPrivate {
        struct network_builder_state_t {
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
        struct network_builder_t {
            using Build = Network<_Data...>;
        };
        template<_NetworkBuilderPrivate::network_builder_state_t _state, typename... _Data>
        static consteval network_builder_t<_state, _Data...> build_network() {
            static_assert(_state.has_float_type, "Network must have a float type!");
            static_assert(_state.layers_satisfied, "All layers must be satisfied!");

            static_assert(
                Allocator::_is_host && (Data::_is_host && ...) ||
                Allocator::_is_device && (Data::_is_device && ...),
                "All elements must be have either host or device support"
            );

            return {};
        }

        template<_NetworkBuilderPrivate::network_builder_state_t _state, typename A, typename FT, typename... _Data>
        struct dataset_builder_t {
            template<typename L1, typename... L>
            [[nodiscard]] static consteval int_t in_size() noexcept { return L1::size; }

            template<typename C, typename... L>
            [[nodiscard]] static consteval int_t out_size2() noexcept {
                if constexpr (sizeof...(L) == 0) return 0;
                else return out_size<L...>();
            }
            template<typename L1, typename... L>
            [[nodiscard]] static consteval int_t out_size() noexcept {
                if constexpr (sizeof...(L) == 0) return L1::size;
                else return out_size2<L...>();
            }

            using Build = Dataset<
                Allocator,
                FloatType,
                sizeof...(_Data) != 0 ? in_size<_Data...>() : 0,
                sizeof...(_Data) != 0 ? out_size<_Data...>() : 0
            >;
        };
        template<_NetworkBuilderPrivate::network_builder_state_t _state, typename A, typename FT>
        struct dataset_builder_t<_state, A, FT> { using Build = Dataset<Allocator, FloatType, 0, 0>; };
        template<_NetworkBuilderPrivate::network_builder_state_t _state, typename... _Data>
        static consteval dataset_builder_t<_state, _Data...> build_dataset() {
            static_assert(_state.has_float_type, "Network must have a float type!");

            static_assert(
                Allocator::_is_host && (Data::_is_host && ...) ||
                Allocator::_is_device && (Data::_is_device && ...),
                "All elements must be have either host or device support"
            );

            return {};
        }

    public:
        using NetworkType = decltype(build_network<state, Allocator, FloatType, Data...>())::Build;
        using DatasetType = decltype(build_dataset<state, Allocator, FloatType, Data...>())::Build;

        template<typename _FloatType> requires(!state.has_float_type)
        using DefineFloatType = _NetworkBuilder<_NetworkBuilderPrivate::network_builder_state_t {
            .has_float_type = true,
            .layers_satisfied = state.layers_satisfied,
            .layer_count = state.layer_count + 1,
        }, Allocator, _FloatType, Data...>;

        template<int_t size>
        using AddLayer = _NetworkBuilder<_NetworkBuilderPrivate::network_builder_state_t {
            .has_float_type = state.has_float_type,
            .layers_satisfied = (state.layer_count > 0),
            .layer_count = state.layer_count + 1,
        }, Allocator, FloatType, Data..., Layer<size>>;

        template<typename Connection>
        using AddConnection = _NetworkBuilder<_NetworkBuilderPrivate::network_builder_state_t {
            .has_float_type = state.has_float_type,
            .layers_satisfied = false,
            .layer_count = state.layer_count,
        }, Allocator, FloatType, Data..., Connection>;
    };

    using NetworkBuilder = _NetworkBuilder<
        _NetworkBuilderPrivate::network_builder_state_t { },
        PKAI::Allocator,
        _NetworkBuilderPrivate::NoneType
    >;
}