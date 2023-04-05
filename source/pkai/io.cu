#include <pkai/network.hpp>

__global__
void give_inputs_k(float ** neurons, float * values, int input_size) {
    float * temp = neurons[0];

    for (int i = 0; i < input_size; i++) temp[i] = values[i];
}

void PKAI::Network::send_inputs(float * values) {
    give_inputs_k<<<1, 1>>>(neurons, values, layer_sizes[0]);
    cudaDeviceSynchronize();
}
void PKAI::Network::send_inputs(float * values, cudaStream_t stream) {
    give_inputs_k<<<1, 1, 0, stream>>>(neurons, values, layer_sizes[0]);
}


__global__
void extract_outputs_k(float ** neurons, float * values, int output_size, int layer_index) {
    cudaMemcpyAsync(
        values,
        neurons[layer_index],
        output_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        0
    );

}

void PKAI::Network::extract_outputs(float * values) {
    extract_outputs_k<<<1, 1>>>(
        neurons,
        values,
        layer_sizes[layer_count - 1],
        layer_count - 1
    );
    cudaDeviceSynchronize();
}