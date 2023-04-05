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

    public:
        const int layer_count;
        int * layer_sizes;

        template<int N>
        inline Network(int (&& _layer_sizes)[N]):
            layer_count(N) {
            layer_sizes = new int[layer_count];
            {
                int * temp = std::move(_layer_sizes);
                for (int i = 0; i < layer_count; i++) layer_sizes[i] = temp[i];
            }

            cudaMalloc((void **) &neurons, layer_count * sizeof(float *));
            cudaMalloc((void **) &synapses, (layer_count - 1) * sizeof(float *));
            cudaMalloc((void **) &costs, (layer_count - 1) * sizeof(float *));
            cudaMalloc((void **) &biases, (layer_count - 1) * sizeof(float *));

            float * temp_neuron_ptrs[layer_count];
            float * temp_synapse_ptrs[layer_count - 1];
            float * temp_correction_ptrs[layer_count - 1];
            float * temp_bias_ptrs[layer_count - 1];

            for (int i = 0; i < layer_count; i++) {
                cudaMalloc((void **) &temp_neuron_ptrs[i], layer_sizes[i] * sizeof(float));
                cudaMemset(temp_neuron_ptrs[i], 0, layer_sizes[i] * sizeof(float));
            }
            for (int i = 0; i < layer_count - 1; i++) {
                cudaMalloc((void **) &temp_synapse_ptrs[i], layer_sizes[i] * layer_sizes[i + 1] * sizeof(float));
                cudaMemset(temp_synapse_ptrs[i], 0, layer_sizes[i] * layer_sizes[i + 1] * sizeof(float));
            }

            for (int i = 1; i < layer_count; i++) {
                cudaMalloc((void **) &temp_correction_ptrs[i - 1], layer_sizes[i] * sizeof(float));
                cudaMemset(temp_correction_ptrs[i - 1], 0, layer_sizes[i] * sizeof(float));
                cudaMalloc((void **) &temp_bias_ptrs[i - 1], layer_sizes[i] * sizeof(float));
                cudaMemset(temp_bias_ptrs[i - 1], 0, layer_sizes[i] * sizeof(float));
            }

//            costs_out = temp_correction_ptrs[layer_count - 2];

            cudaMemcpy(neurons, temp_neuron_ptrs, layer_count * sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(synapses, temp_synapse_ptrs, (layer_count - 1) * sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(costs, temp_correction_ptrs, (layer_count - 1) * sizeof(float *), cudaMemcpyHostToDevice);
            cudaMemcpy(biases, temp_bias_ptrs, (layer_count - 1) * sizeof(float *), cudaMemcpyHostToDevice);
        }

        inline ~Network() {
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

        void send_inputs(float * values);
        void send_inputs(float * values, cudaStream_t stream);
        void extract_outputs(float * values);

        void activate();
        void activate(cudaStream_t stream);
        void backpropagate(float * correct);
        void backpropagate(float * correct, cudaStream_t stream);

        void print();

        void run(TrainingSet & training_set, int iterations, bool draw, bool progress_checks, bool random_values);
    };
}