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
            ::AddConnection<PKAI::Connection::FullyConnected<PKAI::ActivationFunction::ReLu>>
            ::AddLayer<2000>
            ::AddConnection<PKAI::Connection::FullyConnected<PKAI::ActivationFunction::ReLu>>
            ::AddLayer<2000>
            ::AddConnection<PKAI::Connection::FullyConnected<PKAI::ActivationFunction::ReLu>>
            ::AddLayer<2000>
            ::AddConnection<PKAI::Connection::FullyConnected<PKAI::ActivationFunction::ReLu>>
            ::AddLayer<256 * letter_categories>;

        Builder::Network network;
        Builder::Dataset dataset;

        inline void give_string(const char * str) {
            float input[256 * letter_categories] = { 0 };

            for (int i = 0; str[i] != '\0'; i++) {
                for (int j = 0; j < letter_categories; j++) {
                    if (str[i] == ' ') input[i * letter_categories + j] = j == 26;
                    else input[i * letter_categories + j] = j == str[i] - 65;
                }
            }

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
                else out += (char) (largest + 65);
            }

            return out;
        }

    public:
        inline void give_data(const char * in, const char * expected) {
            float input[256 * letter_categories] = { 0 };
            float output[256 * letter_categories] = { 0 };

            {
                for (int i = 0; in[i] != '\0'; i++) {
                    for (int j = 0; j < letter_categories; j++) {
                        if (in[i] == ' ') { if (j == 26) input[i * letter_categories + j] = 1.0f; }
                        else if (j == in[i] - 65) input[i * letter_categories + j] = 1.0f;
                    }
                }
            }

            {
                for (int i = 0; expected[i] != '\0'; i++) {
                    for (int j = 0; j < letter_categories; j++) {
                        if (in[i] == ' ') { if (j == 26) input[i * letter_categories + j] = 1.0f; }
                        else if (j == expected[i] - 65) output[i * letter_categories + j] = 1.0f;
                    }
                }
            }

            dataset.push_set(input, output);
        }

        inline std::string get_response(const char * input) {
            give_string(input);
            network.activate();
            return get_string();
        }

        inline void train() {
            network.template train<10>(dataset, 1000);
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