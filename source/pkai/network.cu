#include <pkai/network.hpp>

void PKAI::Network::extract_network_data(float ** _neurons, float ** _synapses, float ** _biases) {
    float * neuron_ptrs[layer_count];
    float * synapse_ptrs[layer_count - 1];
    float * bias_ptrs[layer_count - 1];
    cudaMemcpy(neuron_ptrs, neurons, layer_count * sizeof(float *), cudaMemcpyDeviceToHost);
    cudaMemcpy(synapse_ptrs, synapses, (layer_count - 1) * sizeof(float *), cudaMemcpyDeviceToHost);
    cudaMemcpy(bias_ptrs, biases, (layer_count - 1) * sizeof(float *), cudaMemcpyDeviceToHost);

    for (int i = 0; i < layer_count; i++) {
        cudaMemcpy(_neurons[i], neuron_ptrs[i], layer_sizes[i] * sizeof(float), cudaMemcpyDeviceToHost);
    }
    for (int i = 0; i < layer_count - 1; i++) {
        cudaMemcpy(_synapses[i], synapse_ptrs[i], layer_sizes[i] * layer_sizes[i + 1] * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(_biases[i], bias_ptrs[i], layer_sizes[i + 1] * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

void PKAI::Network::send_network_data(float ** _neurons, float ** _synapses, float ** _biases) {
    if (_neurons) {
        float * neuron_ptrs[layer_count];
        cudaMemcpy(neuron_ptrs, neurons, layer_count * sizeof(float *), cudaMemcpyDeviceToHost);

        for (int i = 0; i < layer_count; i++) {
            cudaMemcpy(neuron_ptrs[i], _neurons[i], layer_sizes[i] * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    if (_synapses) {
        float * synapse_ptrs[layer_count - 1];
        cudaMemcpy(synapse_ptrs, synapses, (layer_count - 1) * sizeof(float *), cudaMemcpyDeviceToHost);

        for (int i = 0; i < layer_count - 1; i++) {
            cudaMemcpy(synapse_ptrs[i], _synapses[i], layer_sizes[i] * layer_sizes[i + 1] * sizeof(float),
                       cudaMemcpyHostToDevice);
        }
    }

    if (_biases) {
        float * bias_ptrs[layer_count - 1];
        cudaMemcpy(bias_ptrs, biases, (layer_count - 1) * sizeof(float *), cudaMemcpyDeviceToHost);

        for (int i = 0; i < layer_count - 1; i++) cudaMemcpy(
            bias_ptrs[i],
            _biases[i],
            layer_sizes[i + 1] * sizeof(float),
            cudaMemcpyHostToDevice
        );
    }
}