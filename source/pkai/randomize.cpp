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

/*
 * S: 3.06799e+18
S: 7.00649e-45
S: 0
S: 0
S: 0.237478
S: 0.299657
S: 0.32696
S: 0.218569
S: 0.492378
S: 0.400148
S: 0.288102
S: 0.133381
S: -0.271887
S: 0.401795
S: -0.0290827
S: 0.206115
S: 0.180961
S: 0.415074
S: 0.15448
S: 0.342219
S: 3.06805e+18
S: 7.00649e-45

 */