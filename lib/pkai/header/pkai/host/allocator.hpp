#pragma once

#include <cstring>
#include <random>

#include <pkai/host/_host_check.hpp>

#include <pkai/universal/config.hpp>

namespace PKAI {
    struct Allocator {
        template<typename T, int_t n>
        class Instance {
        protected:
            T _data[n];

            std::default_random_engine engine;
            std::uniform_real_distribution<float> distro = std::uniform_real_distribution<float>(0.0001, 0.001);

        public:
            inline Instance() noexcept {
                for (int_t i = 0; i < n; i++) _data[i] = distro(engine);
            }

            inline T * data() noexcept { return _data; }
            [[nodiscard]] inline int_t size() const noexcept { return n; }

            inline void set_data(const T * source, int_t count) noexcept {
                std::memcpy(_data, source, count * sizeof(T));
            }
            inline void get_data(T * dest, int_t count) noexcept {
                std::memcpy(dest, _data, count * sizeof(T));
            }
        };
    };
}