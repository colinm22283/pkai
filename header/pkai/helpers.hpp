#pragma once

#include <cuda_runtime_api.h>

namespace PKAI {
    template<typename T>
    void free_device_2d_array(T ** device_ptr, unsigned long w) {
        float * temp_ptrs[w];

        cudaMemcpy(temp_ptrs, device_ptr, w * sizeof(T *), cudaMemcpyDeviceToHost);

        for (int i = 0; i < w; i++) cudaFree(temp_ptrs[i]);

        cudaFree(device_ptr);
    }
}