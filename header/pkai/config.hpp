#pragma once

namespace PKAI::Config {
    constexpr float learning_rate = 0.1f;
    constexpr unsigned int bp_block_dim = 512;

    constexpr float random_weight_min = -0.5;
    constexpr float random_weight_max = 0.5;
    constexpr float random_bias_min = -0.5;
    constexpr float random_bias_max = 0.5;
}