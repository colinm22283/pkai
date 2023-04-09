#include <iostream>

#include <pkai/network.hpp>

void PKAI::Network::run(TrainingSet & training_set, int iterations, bool draw, bool progress_checks, bool random_values) {
    float average_cost = 0;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < iterations; i++) {
        training_pair_t & pair = random_values ? training_set.get_random_pair() : training_set.get_next_pair();

        send_inputs(pair.inputs(), stream);

        activate(stream);

        backpropagate(pair.outputs(), stream);

        cudaStreamSynchronize(stream);

        if (draw) {
            print();

            int output_size = layer_sizes[layer_count - 1];

            float * last_layer;
            cudaMemcpy(&last_layer, costs + layer_count - 2, sizeof(float *), cudaMemcpyDeviceToHost);
            float errors[output_size];
            cudaMemcpy(errors, last_layer, output_size * sizeof(float), cudaMemcpyDeviceToHost);

            float correct[output_size];
            cudaMemcpy(correct, pair.outputs(), output_size * sizeof(float), cudaMemcpyDeviceToHost);

            std::cout << "Correct:";
            for (int j = 0; j < output_size; j++) {
                std::cout << " " << correct[j];

                average_cost += fabsf32(errors[j] / (float) iterations / (float) output_size);
            }
            std::cout << "\n\n";
        }

        if (progress_checks && i % (iterations / 100) == 0) {
            std::cout << (i * 100 / iterations) << "%\n";
        }
    }

    if (draw) {
        std::cout << "\n";

        std::cout << "Average cost = " << average_cost << "\n";
    }

    cudaStreamDestroy(stream);
}