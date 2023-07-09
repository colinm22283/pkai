#pragma once

#include <pkai/common.hpp>

namespace PKAI {
    template<typename Allocator, typename FloatType>
    class training_pair_t {
        nsize_t _input_size, _output_size;

        Allocator _input;
        Allocator _output;

    public:
        template<std::size_t in, std::size_t on>
        inline training_pair_t(FloatType (&& inputs)[in], FloatType (&& outputs)[on]):
          _input_size(in), _output_size(on),
          _input(std::move(inputs)), _output(std::move(outputs)) { }

        inline training_pair_t(FloatType * inputs, FloatType * outputs, nsize_t in, nsize_t on):
            _input_size(in), _output_size(on),
            _input(inputs, in), _output(outputs, on) { }

        inline training_pair_t(nsize_t __input_size, nsize_t __output_size):
          _input_size(__input_size),
          _output_size(__output_size),
          _input(__input_size),
          _output(__output_size) { }

//        inline training_pair_t(training_pair_t & x):
//          _input_size(x._input_size), _output_size(x._output_size),
//          _input(std::move(x._input)), _output(std::move(x._output));

        [[nodiscard]] inline nsize_t input_size() const noexcept { return _input_size; }
        [[nodiscard]] inline nsize_t output_size() const noexcept { return _output_size; }
        inline FloatType * input() const noexcept { return _input.data; }
        inline FloatType * output() const noexcept { return _output.data; }
    };
}