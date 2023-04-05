#include <random>

#include <pkai/network.hpp>
#include <pkai/config.hpp>

void PKAI::Network::randomize_weights(unsigned long seed) {
    std::default_random_engine random_engine(seed);
    std::uniform_real_distribution<float> distribution(Config::random_weight_min, Config::random_weight_max);

    float * synapses[layer_count - 1];
    for (int i = 0; i < layer_count - 1; i++) {
        synapses[i] = new float[layer_sizes[i] * layer_sizes[i + 1]];
        for (int j = 0; j < layer_sizes[i] * layer_sizes[i + 1]; j++) {
            synapses[i][j] = distribution(random_engine);
        }
    }

    send_network_data(nullptr, synapses, nullptr);

    for (int i = 0; i < layer_count - 1; i++) delete[] synapses[i];
}

void PKAI::Network::randomize_biases(unsigned long seed) {
    std::default_random_engine random_engine(seed);
    std::uniform_real_distribution<float> distribution(Config::random_bias_min, Config::random_bias_max);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float * temp_biases[layer_count - 1];
    cudaMemcpyAsync(temp_biases, biases, (layer_count - 1) * sizeof(float *), cudaMemcpyDeviceToHost, stream);

    for (int i = 0; i < layer_count - 1; i++) {
        for (int j = 0; j < layer_sizes[i + 1]; j++) {
            float r = distribution(random_engine);
            cudaMemcpyAsync(temp_biases[i] + j, &r, sizeof(float), cudaMemcpyHostToDevice, stream);
        }
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

void PKAI::Network::randomize(unsigned long seed) {
    std::default_random_engine random_engine(seed);
    std::uniform_real_distribution<float> distribution(Config::random_weight_min, Config::random_weight_max);

    float * _synapses[layer_count - 1];
    for (int i = 0; i < layer_count - 1; i++) {
        _synapses[i] = new float[layer_sizes[i] * layer_sizes[i + 1]];
        for (int j = 0; j < layer_sizes[i] * layer_sizes[i + 1]; j++) {
            _synapses[i][j] = distribution(random_engine);
        }
    }

    float * _biases[layer_count - 1];
    for (int i = 0; i < layer_count - 1; i++) {
        _biases[i] = new float[layer_sizes[i + 1]];
        for (int j = 0; j < layer_sizes[i + 1]; j++) {
            _biases[i][j] = distribution(random_engine);
        }
    }

    send_network_data(nullptr, _synapses, _biases);

    for (int i = 0; i < layer_count - 1; i++) delete[] _synapses[i];
}