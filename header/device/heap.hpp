#pragma once

#include <algorithm>

#include <cuda_runtime_api.h>

namespace dev {
    template<typename T, typename... Args>
    __global__
    void _alloc_k(T * ptr, Args... args) {
        new (ptr) T(args...);
    }

    template<typename T, typename... Args>
    __host__
    inline T * alloc(Args && ... args) {
        T * temp;
        cudaMalloc((void **) &temp, sizeof(T));
        _alloc_k<<<1, 1>>>(temp, std::forward<Args>(args)...);
        return temp;
    }

    template<typename T>
    __global__
    void _alloc_array_k(T * ptr, std::size_t size) {
        new (ptr) T[size];
    }
    template<typename T>
    __host__
    inline T * alloc_array(std::size_t size) {
        T * temp;
        cudaMalloc((void **) &temp, size * sizeof(T));
        _alloc_array_k<<<1, 1>>>(temp, size);
        return temp;
    }

    template<typename T>
    __global__
    void _free_k(T * ptr) {
        ptr->~T();
    }

    template<typename T>
    __host__
    inline void free(T * ptr) {
        _free_k<<<1, 1>>>(ptr);
        cudaFree(ptr);
    }

    template<typename T>
    __global__
    void _free_array_k(T * ptr, std::size_t size) {
        for (std::size_t i = 0; i < size; i++) ptr[i].~T();
    }
    
    template<typename T>
    __host__
    inline void free_array(T * ptr, std::size_t size) {
        _free_array_k<<<1, 1>>>(ptr, size);
        cudaFree(ptr);
    }

    template<typename PT, typename VT>
    __global__
    void _set_value_k(PT * ptr, VT value) {
        *ptr = value;
    }
    template<typename PT, typename VT>
    __host__
    inline void set_value(PT * ptr, VT value) {
        _set_value_k<<<1, 1>>>(ptr, value);
    }
}