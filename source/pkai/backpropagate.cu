#include <pkai/network.hpp>
#include <pkai/functions.hpp>
#include <pkai/config.hpp>
#include <pkai/indexers.hpp>

__device__
inline float make_nonzero(float x) { x == 0 ? (signbit(x) ? -1.0f : 1.0f) * 1.000000001f : x; }

__device__
inline float error_function(float output, float expected) {
    float temp = (expected - output) * 0.5;

//    return temp;

    return temp * temp * temp;
}

__global__
void copy_corrections(
    float ** neurons,
    float ** costs,
    float * correct,
    unsigned long final_layer,
    unsigned long layer_size
) {
    const unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= layer_size) return;

    PKAI::cost(costs, final_layer, idx) = error_function(
        PKAI::neuron(neurons, final_layer, idx),
        correct[idx]
    );
}

__global__
void backpropagate_k(
    float ** neurons,
    float ** synapses,
    float ** costs,
    float ** biases,
    unsigned long layer,
    unsigned long layer_size,
    unsigned long to_size
) {
    const unsigned long idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= layer_size) return;

    // starts on second to last layer

    if (layer == 0) {
        for (int i = 0; i < to_size; i++) {
            PKAI::synapse(synapses, layer, layer_size, idx, i) +=
                PKAI::cost(costs, layer + 1, i) * // to cost
                PKAI::neuron(neurons, layer, idx) * // from neuron
                PKAI::Config::learning_rate;

            PKAI::bias(biases, layer + 1, i) +=
                PKAI::cost(costs, layer + 1, i) * // to cost
                PKAI::Config::learning_rate;
        }
    } else {
        PKAI::cost(costs, layer, idx) = 0;

        for (int i = 0; i < to_size; i++) {
            PKAI::synapse(synapses, layer, layer_size, idx, i) +=
                PKAI::cost(costs, layer + 1, i) * // to cost
                PKAI::neuron(neurons, layer, idx) * // from neuron
                PKAI::Config::learning_rate;

            PKAI::bias(biases, layer + 1, i) +=
                PKAI::cost(costs, layer + 1, i) * // to cost
                PKAI::Config::learning_rate;

            PKAI::cost(costs, layer, idx) -=
                PKAI::cost(costs, layer + 1, i) * // to cost
                PKAI::synapse(synapses, layer, layer_size, idx, i); // intermediate weight
        }
    }
}

void PKAI::Network::backpropagate(float * correct) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    copy_corrections<<<layer_sizes[layer_count - 1] / PKAI::Config::bp_block_dim + 1, PKAI::Config::bp_block_dim, 0, stream>>>(
        neurons,
        costs,
        correct,
        layer_count - 1,
        layer_sizes[layer_count - 1]
    );

    for (int i = layer_count - 2; i >= 0; i--) {
        backpropagate_k<<<layer_sizes[i] / PKAI::Config::bp_block_dim + 1, PKAI::Config::bp_block_dim, 0, stream>>>(
            neurons,
            synapses,
            costs,
            biases,
            i,
            layer_sizes[i],
            layer_sizes[i + 1]
        );
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

void PKAI::Network::backpropagate(float * correct, cudaStream_t stream) {
    copy_corrections<<<layer_sizes[layer_count - 1] / PKAI::Config::bp_block_dim + 1, PKAI::Config::bp_block_dim, 0, stream>>>(
        neurons,
        costs,
        correct,
        layer_count - 1,
        layer_sizes[layer_count - 1]
    );

    for (int i = layer_count - 2; i >= 0; i--) {
        backpropagate_k<<<layer_sizes[i] / PKAI::Config::bp_block_dim + 1, PKAI::Config::bp_block_dim, 0, stream>>>(
            neurons,
            synapses,
            costs,
            biases,
            i,
            layer_sizes[i],
            layer_sizes[i + 1]
        );
    }
}