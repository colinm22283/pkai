#pragma once

#include <cuda_runtime_api.h>

#include <pkai/training_set.hpp>
#include <pkai/helpers.hpp>

namespace PKAI {
    class Network {
    protected:
        float ** neurons;
        float ** synapses;
        float ** costs;
        float ** biases;

        float * output_layer;

        inline void device_allocate() {
            cudaMalloc((void **) &neurons, layer_count * sizeof(float *));
            cudaMalloc((void **) &synapses, (layer_count - 1) * sizeof(float *));
            cudaMalloc((void **) &costs, (layer_count - 1) * sizeof(float *));
            cudaMalloc((void **) &biases, (layer_count - 1) * sizeof(float *));

            float * temp_neuron_ptrs[layer_count];
            float * temp_synapse_ptrs[layer_count - 1];
            float * temp_cost_ptrs[layer_count - 1];
            float * temp_bias_ptrs[layer_count - 1];

            for (int i = 0; i < layer_count; i++) {
                cudaMalloc((void **) &temp_neuron_ptrs[i], layer_sizes[i] * sizeof(float));
            }
            for (int i = 0; i < layer_count - 1; i++) {
                cudaMalloc((void **) &temp_synapse_ptrs[i], layer_sizes[i] * layer_sizes[i + 1] * sizeof(float));

                cudaMalloc((void **) &temp_cost_ptrs[i], layer_sizes[i + 1] * sizeof(float));

                cudaMalloc((void **) &temp_bias_ptrs[i], layer_sizes[i + 1] * sizeof(float));
            }

            output_layer = temp_neuron_ptrs[layer_count - 1];

            cudaMemcpy(neurons, temp_neuron_ptrs, layer_count * sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(synapses, temp_synapse_ptrs, (layer_count - 1) * sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(costs, temp_cost_ptrs, (layer_count - 1) * sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(biases, temp_bias_ptrs, (layer_count - 1) * sizeof(float *), cudaMemcpyHostToDevice);
        }

    public:
        unsigned long layer_count;
        unsigned long * layer_sizes;

        template<unsigned long N>
        Network(unsigned long (&& _layer_sizes)[N]):
            Network(std::move(_layer_sizes), N) { }

        inline Network(const unsigned long * _layer_sizes, unsigned long _layer_count):
          layer_count(_layer_count) {
            layer_sizes = new unsigned long[layer_count];
            for (unsigned long i = 0; i < layer_count; i++) layer_sizes[i] = _layer_sizes[i];

            device_allocate();
        }
        Network(const char * path);

        Network(Network &) = delete;
        Network(Network &&) = delete;

        inline ~Network() {
            std::cout << "Dealloc\n";
            free_device_2d_array(neurons, layer_count);
            free_device_2d_array(synapses, layer_count - 1);
            free_device_2d_array(costs, layer_count - 1);
            free_device_2d_array(biases, layer_count - 1);

            delete[] layer_sizes;
        }

        void randomize_weights(unsigned long seed);
        void randomize_biases(unsigned long seed);
        void randomize(unsigned long seed);

        void extract_network_data(float ** _neurons, float ** _synapses, float ** biases);
        void send_network_data(float ** _neurons, float ** _synapses, float ** biases);

        void send_inputs(const float * values);
        void send_inputs(const float * values, cudaStream_t stream);
        void extract_outputs(float * values);
        void extract_outputs(float * values, cudaStream_t stream);

        void activate();
        void activate(cudaStream_t stream);
        void backpropagate(const float * correct);
        void backpropagate(const float * correct, cudaStream_t stream);

        void print();

        void run(TrainingSet & training_set, int iterations, bool draw, bool progress_checks, bool random_values);

        void save(const char * path);
    };
}