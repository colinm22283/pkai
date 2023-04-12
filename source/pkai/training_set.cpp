#include <fstream>
#include <stdexcept>

#include <pkai/training_set.hpp>

PKAI::TrainingSet::TrainingSet(const char * path) {
    std::ifstream fs(path);

    if (!fs.is_open()) throw std::runtime_error("File not found");

    fs.read((char *) &input_size, sizeof(unsigned long));
    fs.read((char *) &output_size, sizeof(unsigned long));

    float inputs[input_size];
    float outputs[output_size];

    while (true) {
        if (!fs.read((char *) inputs, input_size * sizeof(float))) break;;
        if (!fs.read((char *) outputs, output_size * sizeof(float))) break;

        if (fs.eof()) break;

        this->add(PKAI::training_pair_t(
            inputs,
            input_size,
            outputs,
            output_size
        ));
    }
}

void PKAI::TrainingSet::save(const char * path) {
    std::ofstream fs(path);

    fs.write((char *) &input_size, sizeof(unsigned long));
    fs.write((char *) &output_size, sizeof(unsigned long));

    float inputs[input_size];
    float outputs[output_size];

    for (training_pair_t & tp : pairs) {
        cudaMemcpy(
            inputs,
            tp.inputs(),
            input_size * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        cudaMemcpy(
            outputs,
            tp.outputs(),
            output_size * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        fs.write((char *) &inputs, (long) input_size * sizeof(float));
        fs.write((char *) &outputs, (long) output_size * sizeof(float));
    }

    fs.close();
}