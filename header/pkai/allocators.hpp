#pragma once

namespace PKAI {
    template<typename T>
    struct _host_allocator_t {
        T * data;

        template<nsize_t n>
        explicit inline _host_allocator_t(T (&& _data)[n]): data(new T[n]) {
            T * temp = std::move(_data);

            for (nsize_t i = 0; i < n; i++) data[i] = temp[i];
        }
        inline _host_allocator_t(T * _data, nsize_t n): data(new T[n]) {
            for (nsize_t i = 0; i < n; i++) data[i] = _data[i];
        }
        explicit inline _host_allocator_t(nsize_t size): data(new T[size]) { }
        inline ~_host_allocator_t() { delete[] data; }

        inline _host_allocator_t(_host_allocator_t &) = delete;
        inline _host_allocator_t(_host_allocator_t && x): data(x.data) { x.data = nullptr; }
    };
}