#include <fstream>

#include <pkai/training_set.hpp>

PKAI::TrainingSet::TrainingSet(const char * path) {
    std::ifstream fs(path);

    fs.read((char *) &input_size, sizeof(unsigned long));
    fs.read((char *) &output_size, sizeof(unsigned long));

    float inputs[input_size];
    float outputs[output_size];

    while (true) {
        fs.read((char *) inputs, input_size * sizeof(float));
        fs.read((char *) outputs, output_size * sizeof(float));

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
            tp.inputs,
            input_size * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        cudaMemcpy(
            outputs,
            tp.outputs,
            output_size * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        fs.write((char *) &inputs, input_size * sizeof(float));
        fs.write((char *) &outputs, output_size * sizeof(float));
    }

    fs.close();
}