#pragma once

#include <cuda_runtime.h>

#include "../../device/device_new.hpp"

#include "../../config.hpp"

namespace PKAI::Allocators {
    struct DeviceAllocator {
        static constexpr bool _is_allocator = true;
        static constexpr bool _is_host = false;
        static constexpr bool _is_device = true;

        template<typename T, int_t _size>
        class Allocate {
        protected:
            T * _data;

        public:
            inline Allocate() { cudaMalloc((void **) &_data, _size * sizeof(T)); }
            inline ~Allocate() { cudaFree(_data); }

            inline void set_data(T const * data, int_t count) {
                cudaMemcpy(_data, data, count * sizeof(T), cudaMemcpyHostToDevice);
            }
            inline void get_data(T const * data, int_t count) {
                cudaMemcpy(data, _data, count * sizeof(T), cudaMemcpyDeviceToHost);
            }

            inline T * data() const noexcept { return _data; }
            [[nodiscard]] consteval int_t size() const noexcept { return _size; }
        };
    };
}