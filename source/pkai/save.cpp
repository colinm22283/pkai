#include <fstream>

#include <cuda_device_runtime_api.h>

#include <pkai/network.hpp>

PKAI::Network::Network(const char * path) {
    std::ifstream file(path);

    file.read((char *) &layer_count, sizeof(unsigned long));
    layer_sizes = new unsigned long[layer_count];
    for (int i = 0; i < layer_count; i++) {
        file.read((char *) &layer_sizes[i], sizeof(unsigned long));
    }

    float ** temp_synapses = new float *[layer_count - 1];
    float ** temp_biases = new float *[layer_count - 1];
    for (unsigned long i = 0; i < layer_count - 1; i++) {
        temp_synapses[i] = new float[layer_sizes[i] * layer_sizes[i + 1]];
        temp_biases[i] = new float[layer_sizes[i + 1]];
    }

    for (unsigned long i = 0; i < layer_count - 1; i++) {
        for (unsigned long j = 0; j < layer_sizes[i] * layer_sizes[i + 1]; j++) {
            file.read((char *) &temp_synapses[i][j], sizeof(float));
        }
    }
    for (unsigned long i = 0; i < layer_count - 1; i++) {
        for (unsigned long j = 0; j < layer_sizes[i + 1]; j++) {
            file.read((char *) &temp_biases[i][j], sizeof(float));
        }
    }

    device_allocate();

    send_network_data(nullptr, temp_synapses, temp_biases);

    file.close();

    for (unsigned long i = 0; i < layer_count - 1; i++) {
        delete[] temp_synapses[i]; delete[] temp_biases[i];
    }
    delete[] temp_synapses; delete[] temp_biases;
}

void PKAI::Network::save(const char * path) {
    std::ofstream file(path, std::ios::trunc);

    file.write((char *) &layer_count, sizeof(unsigned long));
    for (int i = 0; i < layer_count; i++) {
        file.write((char *) &layer_sizes[i], sizeof(unsigned long));
    }

    float ** temp_synapses = new float *[layer_count - 1];
    float ** temp_biases = new float *[layer_count - 1];
    for (unsigned long i = 0; i < layer_count - 1; i++) {
        temp_synapses[i] = new float[layer_sizes[i] * layer_sizes[i + 1]];
        temp_biases[i] = new float[layer_sizes[i + 1]];
    }

    extract_network_data(nullptr, temp_synapses, temp_biases);

    for (unsigned long i = 0; i < layer_count - 1; i++) {
        for (unsigned long j = 0; j < layer_sizes[i] * layer_sizes[i + 1]; j++) {
            file.write((char *) &temp_synapses[i][j], sizeof(float));
        }

    }

    for (unsigned long i = 0; i < layer_count - 1; i++) {
        for (unsigned long j = 0; j < layer_sizes[i + 1]; j++) {
            file.write((char *) &temp_biases[i][j], sizeof(float));
        }
    }

    file.close();
}