#pragma once

#include <pkai/host.hpp>
#include <pkai/universal/network_builder.hpp>
#include <pkai/universal/activation_function/relu.hpp>
#include <pkai/universal/connection/fully_connected.hpp>

namespace PKLM {
    class Model {
    protected:
        static constexpr int letter_categories = 30;

        using Builder = PKAI::NetworkBuilder
            ::DefineFloatType<float>
            ::AddLayer<256 * letter_categories>
//            ::AddConnection<PKAI::Connection::FullyConnected<PKAI::ActivationFunction::ReLu>>
//            ::AddLayer<2000>
//            ::AddConnection<PKAI::Connection::FullyConnected<PKAI::ActivationFunction::ReLu>>
//            ::AddLayer<2000>
            ::AddConnection<PKAI::Connection::FullyConnected<PKAI::ActivationFunction::ReLu>>
            ::AddLayer<400>
            ::AddConnection<PKAI::Connection::FullyConnected<PKAI::ActivationFunction::ReLu>>
            ::AddLayer<400>
            ::AddConnection<PKAI::Connection::FullyConnected<PKAI::ActivationFunction::ReLu>>
            ::AddLayer<256 * letter_categories>;

        Builder::Network network;
        Builder::Dataset dataset;

        inline void give_string(const char * str) {
            float input[256 * letter_categories] = { 0 };
            std::memset(input, 0, 256 * letter_categories * sizeof(float));

            int i;
            for (i = 0; str[i] != '\0'; i++) {
                if (str[i] == ' ') { input[i * letter_categories + 26] = 1.0f; }
                else input[i * letter_categories + str[i] - 65] = 1.0f;
            }
            input[i * letter_categories + 27] = 1.0f;

            network.set_inputs(input);
        }
        inline std::string get_string() {
            std::string out;

            float output[256 * letter_categories] = { 0 };
            network.get_outputs(output);

            for (int i = 0; i < 256 * letter_categories; i += letter_categories) {
                int largest = 0;
                for (int j = 1; j < letter_categories; j++) if (output[i + j] > output[i + largest]) largest = j;
                if (largest == 26) out += ' ';
                else if (largest == 27) break;
                else out += (char) (largest + 65);
            }

            return out;
        }

    public:
        inline void give_data(const char * in, const char * expected) {
            float input[256 * letter_categories] = { 0 };
            float output[256 * letter_categories] = { 0 };
            std::memset(input, 0, 256 * letter_categories * sizeof(float));
            std::memset(output, 0, 256 * letter_categories * sizeof(float));

            {
                int i;
                for (i = 0; in[i] != '\0'; i++) {
                    if (in[i] == ' ') input[i * letter_categories + 26] = 1.0f;
                    else input[i * letter_categories + in[i] - 65] = 1.0f;
                }
                input[i * letter_categories + 27] = 1.0f;
            }

            {
                int i;
                for (i = 0; expected[i] != '\0'; i++) {
                    if (expected[i] == ' ') output[i * letter_categories + 26] = 1.0f;
                    else output[i * letter_categories + expected[i] - 65] = 1.0f;
                }
                output[i * letter_categories + 27] = 1.0f;
            }

            dataset.push_set(input, output);
        }

        inline std::string get_response(const char * input) {
            give_string(input);
            network.activate();
            return get_string();
        }

        inline void train(PKAI::int_t iterations) {
            network.template train<10>(dataset, iterations);
        }
        inline float cost() {
            return network.total_cost(dataset);
        }

        inline void save(const char * path) {
            network.save(path);
        }
        inline void load(const char * path) {
            network.load(path);
        }
    };
}