#pragma once

#include <cuda_runtime.h>

#include "pkai/universal/config.hpp"

namespace PKAI {
    struct Allocator {
        template<typename T, int_t n>
        class Instance {
        protected:
            T * _data;

        public:
            inline Instance() { cudaMalloc(&_data, n * sizeof(T)); }
            inline ~Instance() { cudaFree(_data); }

            consteval int_t size() const noexcept { return n; }

            inline void set_data(T * const source, int_t count) noexcept {
//                std::memcpy(_data, source, count * sizeof(T));
            }
            inline void get_data(T * const dest, int_t count) noexcept {
//                std::memcpy(dest, _data, count * sizeof(T));
            }
        };
    };
}