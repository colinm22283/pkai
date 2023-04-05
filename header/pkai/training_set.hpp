#pragma once

#include <vector>
#include <random>
#include <iostream>

#include <cuda_runtime_api.h>

namespace PKAI {
    class training_pair_t {
    protected:
    public: // TODO: remove
        float * inputs;
        float * outputs;

    public:
        consteval training_pair_t():
          inputs(nullptr), outputs(nullptr) { }

        template<int input_size, int output_size>
        inline training_pair_t(float (&& _inputs_ref)[input_size], float (&& _outputs_ref)[output_size]):
          training_pair_t(std::move(_inputs_ref), input_size, std::move(_outputs_ref), output_size) { }

        inline training_pair_t(float * _inputs, unsigned long input_size, float * _outputs, unsigned long output_size) {
            cudaMalloc((void **) &inputs, input_size * sizeof(float));
            cudaMalloc((void **) &outputs, output_size * sizeof(float));

            cudaMemcpy(
                inputs,
                _inputs,
                input_size * sizeof(float),
                cudaMemcpyHostToDevice
            );
            cudaMemcpy(
                outputs,
                _outputs,
                output_size * sizeof(float),
                cudaMemcpyHostToDevice
            );
        }

        training_pair_t(training_pair_t &) = delete;
        inline training_pair_t(training_pair_t && old) noexcept:
          inputs(old.inputs), outputs(old.outputs) {
            old.inputs = nullptr;
            old.outputs = nullptr;
        }

        training_pair_t & operator=(training_pair_t &) = delete;
        inline training_pair_t & operator=(training_pair_t && old) noexcept {
            inputs = old.inputs;
            outputs = old.outputs;

            old.inputs = nullptr;
            old.outputs = nullptr;

            return *this;
        }

        inline ~training_pair_t() {
            if (inputs) cudaFree(inputs);
            if (outputs) cudaFree(outputs);
        }
    };

    class TrainingSet {
    protected:
        static constexpr unsigned long random_seed = 93244037282;

        unsigned long input_size, output_size;
        std::vector<training_pair_t> pairs;
        unsigned long current_pair = 0;

        std::default_random_engine random_engine;
        std::uniform_int_distribution<std::size_t> random_distribution;

    public:
        inline TrainingSet(unsigned long _input_size, unsigned long _output_size):
          input_size(_input_size),
          output_size(_output_size),
          random_engine(random_seed) { }

        template<std::size_t n>
        inline TrainingSet(unsigned long _input_size, unsigned long _output_size, training_pair_t (&& _pairs)[n]):
          input_size(_input_size),
          output_size(_output_size),
          pairs(n),
          random_engine(random_seed) {
            training_pair_t * temp = std::move(_pairs);
            for (int i = 0; i < n; i++) pairs[i] = std::forward<training_pair_t>(temp[i]);
            random_distribution = std::uniform_int_distribution<std::size_t>(0, n - 1);
        }

        TrainingSet(const char * path);

        void save(const char * path);

        inline void add(training_pair_t && pair) {
            random_distribution = std::uniform_int_distribution<std::size_t>(0, pairs.size());
            pairs.push_back(std::move(pair));
        }

        /// \returns A random reference to a device pair
        inline training_pair_t & get_random_pair() {
            return pairs[random_distribution(random_engine)];
        }

        /// \returns The next reference to a device pair
        inline training_pair_t & get_next_pair() {
            unsigned long temp = current_pair++;

            if (current_pair >= pairs.size()) current_pair = 0;

            return pairs[temp];
        }

        inline std::size_t size() const noexcept { return pairs.size(); }
    };
}