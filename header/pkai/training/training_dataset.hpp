#pragma once

#include <vector>
#include <random>

#include <pkai/allocators.hpp>

#include <pkai/training/training_pair.hpp>

namespace PKAI {
    template<typename Allocator, typename FloatType>
    class TrainingDataset {
    protected:
        nsize_t _input_size, _output_size;

        std::vector<training_pair_t<Allocator, FloatType>> pairs;

    public:
        TrainingDataset(nsize_t __input_size, nsize_t __output_size):
          _input_size(__input_size), _output_size(__output_size) { }

        template<std::size_t in, std::size_t on>
        inline void add(FloatType (&& inputs)[in], FloatType (&& outputs)[on]) {
            pairs.emplace_back(std::move(inputs), std::move(outputs));
        }

        inline training_pair_t<Allocator, FloatType> const & get_random_pair() const noexcept {
            std::uniform_int_distribution<std::size_t> distro(0, (int) pairs.size() - 1);

            return pairs[distro(random_generator)];
        }

        [[nodiscard]] inline nsize_t input_size() const noexcept { return _input_size; }
        [[nodiscard]] inline nsize_t output_size() const noexcept { return _output_size; }
    };

    template<typename FloatType>
    using HostDataset = TrainingDataset<_host_allocator_t<FloatType>, FloatType>;
}