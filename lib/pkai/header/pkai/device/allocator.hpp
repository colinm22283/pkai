#pragma once

#include <random>

#include <cuda_runtime.h>

#include "pkai/universal/config.hpp"

namespace PKAI {
    struct Allocator {
        template<typename T, int_t n>
        class Instance {
        protected:
            T * _data;

            std::default_random_engine engine;
            std::uniform_real_distribution<float> distro = std::uniform_real_distribution<float>(0.0001, 0.001);

        public:
            inline Instance() noexcept {
                cudaMalloc((void **) &_data, n * sizeof(T));

                T temp[n];
                for (int_t i = 0; i < n; i++) temp[i] = distro(engine);
                set_data(temp, n);
            }

            inline T * data() noexcept { return _data; }
            [[nodiscard]] inline int_t size() const noexcept { return n; }

            inline void set_data(const T * source, int_t count) noexcept {
                cudaMemcpy(_data, source, count * sizeof(T), cudaMemcpyHostToDevice);
            }
            inline void get_data(T * dest, int_t count) noexcept {
                cudaMemcpy(dest, _data, count * sizeof(T), cudaMemcpyDeviceToHost);
            }
        };
    };
}