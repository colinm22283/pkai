#include <pkai/network.hpp>

__global__
void give_inputs_k(float ** neurons, const float * values, int input_size) {
    float * temp = neurons[0];

    for (int i = 0; i < input_size; i++) temp[i] = values[i];
}

/// @param values Must be a device pointer
void PKAI::Network::send_inputs(const float * values) {
    give_inputs_k<<<1, 1>>>(neurons, values, layer_sizes[0]);
    cudaDeviceSynchronize();
}
/// @param values Must be a device pointer
void PKAI::Network::send_inputs(const float * values, cudaStream_t stream) {
    give_inputs_k<<<1, 1, 0, stream>>>(neurons, values, layer_sizes[0]);
}

/// @param values Must be a host pointer
void PKAI::Network::extract_outputs(float * values) {
    cudaMemcpy(
        values,
        output_layer,
        layer_sizes[layer_count - 1] * sizeof(float),
        cudaMemcpyDeviceToHost
    );
}
/// @param values Must be a host pointer
void PKAI::Network::extract_outputs(float * values, cudaStream_t stream) {
    cudaMemcpyAsync(
        values,
        output_layer,
        layer_sizes[layer_count - 1] * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream
    );
}