#include <iostream>

#include <pkai/network.hpp>

void PKAI::Network::print() {
    float ** _neurons = new float *[layer_count];
    for (int i = 0; i < layer_count; i++) {
        _neurons[i] = new float[layer_sizes[i]];
    }
    float ** _synapses = new float *[layer_count - 1];
    float ** _biases = new float *[layer_count - 1];
    for (int i = 0; i < layer_count - 1; i++) {
        _synapses[i] = new float[layer_sizes[i] * layer_sizes[i + 1]];
        _biases[i] = new float[layer_sizes[i + 1]];
    }

    extract_network_data(_neurons, _synapses, _biases);

    for (int i = 0; i < layer_count; i++) {
        if (i != 0) {
            std::cout << "Biases[" << i + 1 << "]: ";
            for (int j = 0; j < layer_sizes[i]; j++) {
                std::cout << _biases[i - 1][j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "Layer[" << i + 1 << "]:  ";
        for (int j = 0; j < layer_sizes[i]; j++) {
            std::cout << _neurons[i][j] << " ";
        }
        std::cout << "\n";

        if (i < layer_count - 1) {
            std::cout << "Synapses[" << i + 1 << "-" << i + 2 << "]: ";

            for (int j = 0; j < layer_sizes[i] * layer_sizes[i + 1]; j++) {
                std::cout << _synapses[i][j] << " ";
            }
        }

        std::cout << "\n";
    }

    for (int i = 0; i < layer_count; i++) delete[] _neurons[i];
    delete[] _neurons;
    for (int i = 0; i < layer_count - 1; i++) delete[] _synapses[i];
    delete[] _synapses;
}