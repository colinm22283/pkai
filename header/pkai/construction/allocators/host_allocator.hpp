#pragma once

#include <cstring>

#include "../../config.hpp"

namespace PKAI::Allocators {
    struct HostAllocator {
        static constexpr bool _is_allocator = true;
        static constexpr bool _is_host = true;
        static constexpr bool _is_device = false;

        template<typename T, int_t _size>
        class Allocate {
        protected:
            T _data[_size];

        public:
            inline void set_data(T const * data, int_t count) {
                std::memcpy((void *) _data, data, count * sizeof(T));
            }
            inline void get_data(T const * data, int_t count) {
                std::memcpy((void *) data, _data, count * sizeof(T));
            }

            inline T * data() noexcept { return _data; }
            [[nodiscard]] consteval int_t size() const noexcept { return _size; }
        };
    };
}