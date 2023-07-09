#pragma once

#include <pkai/common.hpp>

#include <device/heap.hpp>

namespace PKAI {
    template<lsize_t _size>
    struct Layer {
        static constexpr lsize_t size = _size;
        static constexpr bool is_layer = true;

        template<typename VT>
        struct Config {
            static constexpr lsize_t size = _size;
            static constexpr bool is_layer = true;

            VT * neurons;

            Config(): neurons(dev::alloc_array<VT>(size)) { }
            ~Config() { dev::free_array(neurons, size); }
        };
    };
}