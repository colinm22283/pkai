#pragma once

#include <vector>
#include <fstream>
#include <exception>
#include <random>

#include <pkai/universal/config.hpp>

namespace PKAI {
    template<typename Allocator, typename FloatType, int_t _in_size, int_t _out_size>
    class Dataset {
    protected:
        class IOSet {
        protected:
            Allocator::template Instance<FloatType, _in_size> _in;
            Allocator::template Instance<FloatType, _in_size> _out;

        public:
            template<int_t ni, int_t no>
            inline IOSet(FloatType (&& in)[ni], FloatType (&& out)[no]) noexcept {
                static_assert(ni == _in_size && no == _out_size, "Data sizes don't match");
                _in.set_data(in, _in_size);
                _out.set_data(out, _out_size);
            }
            inline IOSet(FloatType * in, FloatType * out) noexcept {
                _in.set_data(in, _in_size);
                _out.set_data(out, _out_size);
            }

            [[nodiscard]] inline int_t in_size() const noexcept { return _in_size; }
            [[nodiscard]] inline int_t out_size() const noexcept { return _out_size; }
            inline FloatType * in() noexcept { return _in.data(); }
            inline FloatType * out() noexcept { return _out.data(); }
        };

        std::vector<IOSet> data;

    public:
        inline Dataset() noexcept = default;
        explicit inline Dataset(const char * path) {
            std::ifstream fs(path);

            if (!fs.is_open()) throw std::runtime_error("Unable to open file");

            uint32_t size;
            fs.read((char *) &size, sizeof(uint32_t));

            for (int i = 0; i < size; i++) {
                FloatType in[_in_size];
                FloatType out[_out_size];
                fs.read((char *) in, _in_size * sizeof(FloatType));
                fs.read((char *) out, _out_size * sizeof(FloatType));

                if (fs.bad() || fs.eof()) throw std::runtime_error("Corrupt dataset file");

                data.emplace_back(in, out);
            }
        }

        [[nodiscard]] inline int_t size() const noexcept { return data.size(); }
        inline IOSet & get(int_t index) noexcept { return data[index]; }
        inline IOSet & get_random(int seed) noexcept {
            std::default_random_engine engine(seed);
            std::uniform_int_distribution distro(0, size() - 1);
            return data[distro(engine)];
        }
        inline IOSet & operator[](int_t index) noexcept { return data[index]; }

        template<int_t ni, int_t no>
        inline auto & emplace_set(FloatType (&& in)[ni], FloatType (&& out)[no]) {
            data.emplace_back(std::move(in), std::move(out));
            return *this;
        }
        inline auto & push_set(FloatType * in, FloatType * out) {
            data.emplace_back(in, out);
            return *this;
        }

        inline void save(const char * path) noexcept {
            std::ofstream fs(path, std::ofstream::trunc);

            uint32_t size = data.size();
            fs.write((char *) &size, sizeof(uint32_t));

            for (auto & ele : data) {
                fs.write((char *) ele.in(), ele.in_size() * sizeof(FloatType));
                fs.write((char *) ele.out(), ele.out_size() * sizeof(FloatType));
            }
        }
    };
}