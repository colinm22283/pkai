#include <iostream>

#include <pkai/network.hpp>

void PKAI::Network::run(TrainingSet & training_set, unsigned long iterations) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < iterations; i++) {
        training_pair_t & pair = training_set.get_next_pair();

        send_inputs(pair.inputs(), stream);

        activate(stream);

        backpropagate(pair.outputs(), stream);

        cudaStreamSynchronize(stream);
    }

    cudaStreamDestroy(stream);
}

void PKAI::Network::run(
    TrainingSet & training_set,
    unsigned long iterations,
    unsigned long progress_interval,
    void(progress_callback)(Network & network, unsigned long current_iteration)
) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < iterations; i++) {
        training_pair_t & pair = training_set.get_next_pair();

        send_inputs(pair.inputs(), stream);

        activate(stream);

        backpropagate(pair.outputs(), stream);

        cudaStreamSynchronize(stream);

        if (i % progress_interval == 0) progress_callback(*this, i);
    }

    cudaStreamDestroy(stream);
}

void PKAI::Network::run(
    TrainingSet & training_set,
    unsigned long iterations,
    void(progress_callback)(Network & network, training_pair_t & current_pair, unsigned long current_iteration)
) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < iterations; i++) {
        training_pair_t & pair = training_set.get_next_pair();

        send_inputs(pair.inputs(), stream);

        activate(stream);

        backpropagate(pair.outputs(), stream);

        cudaStreamSynchronize(stream);

        progress_callback(*this, pair, i);
    }

    cudaStreamDestroy(stream);
}