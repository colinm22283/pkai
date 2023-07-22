#pragma once

#include <vector>
#include <random>
#include <fstream>

#include <pkai/allocators.hpp>

#include <pkai/training/training_pair.hpp>

namespace PKAI {
    template<typename Allocator, typename FloatType>
    class TrainingDataset {
    protected:
        nsize_t _input_size, _output_size;

        std::vector<training_pair_t<Allocator, FloatType>> pairs;

    public:
        inline TrainingDataset(nsize_t __input_size, nsize_t __output_size):
          _input_size(__input_size), _output_size(__output_size) { }

        explicit inline TrainingDataset(const char * path) {
            std::ifstream fs(path);

            if (!fs.is_open()) throw std::runtime_error(std::string("Unable to locate file ") + path);

            if (!fs.read((char *) &_input_size, sizeof(_input_size))) throw std::runtime_error("Invalid file data");
            if (!fs.read((char *) &_output_size, sizeof(_output_size))) throw std::runtime_error("Invalid file data");

            while (!fs.eof()) {
                FloatType in_buf[_input_size];
                FloatType out_buf[_output_size];

                if (!fs.read((char *) in_buf, _input_size * sizeof(FloatType))) break;
                if (!fs.read((char *) out_buf, _output_size * sizeof(FloatType))) break;

                add(in_buf, out_buf);
            }
        }

        template<std::size_t in, std::size_t on>
        inline void add(FloatType (&& inputs)[in], FloatType (&& outputs)[on]) {
            pairs.emplace_back(std::move(inputs), std::move(outputs));
        }
        inline void add(FloatType * inputs, FloatType * outputs) {
            pairs.emplace_back(inputs, outputs, _input_size, _output_size);
        }

        [[nodiscard]] inline std::size_t size() const noexcept { return pairs.size(); }

        inline training_pair_t<Allocator, FloatType> const & get_random_pair() const noexcept {
            std::uniform_int_distribution<std::size_t> distro(0, (int) pairs.size() - 1);

            return pairs[distro(random_generator)];
        }

        inline void save_to(const char * path) {
            std::ofstream fs(path, std::ofstream::trunc);

            fs.write((char *) & _input_size, sizeof(_input_size));
            fs.write((char *) & _output_size, sizeof(_output_size));
            for (std::size_t i = 0; i < pairs.size(); i++) {
                fs.write((char *) pairs[i].input(), _input_size * sizeof(FloatType));
                fs.write((char *) pairs[i].output(), _output_size * sizeof(FloatType));
            }
        }

        [[nodiscard]] inline nsize_t input_size() const noexcept { return _input_size; }
        [[nodiscard]] inline nsize_t output_size() const noexcept { return _output_size; }
    };

    template<typename FloatType>
    using HostDataset = TrainingDataset<_host_allocator_t<FloatType>, FloatType>;
}