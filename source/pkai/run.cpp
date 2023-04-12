#include <iostream>

#include <pkai/network.hpp>

void PKAI::Network::run(TrainingSet & training_set, unsigned long iterations) {
    for (unsigned long i = 0; i < iterations; i++) {
        training_pair_t & pair = training_set.get_random_pair();

        send_inputs(pair.inputs());

        activate();

        backpropagate(pair.outputs());
    }
}

void PKAI::Network::run(
    TrainingSet & training_set,
    unsigned long iterations,
    unsigned long progress_interval,
    void(progress_callback)(Network & network, unsigned long current_iteration)
) {
    for (unsigned long i = 0; i < iterations; i++) {
        training_pair_t & pair = training_set.get_random_pair();

        send_inputs(pair.inputs());

        activate();

        backpropagate(pair.outputs());

        if (i % progress_interval == 0) progress_callback(*this, i);
    }
}

void PKAI::Network::run(
    TrainingSet & training_set,
    unsigned long iterations,
    void(progress_callback)(Network & network, training_pair_t & current_pair, unsigned long current_iteration)
) {
    for (unsigned long i = 0; i < iterations; i++) {
        training_pair_t & pair = training_set.get_random_pair();

        send_inputs(pair.inputs());

        activate();

        backpropagate(pair.outputs());

        progress_callback(*this, pair, i);
    }
}