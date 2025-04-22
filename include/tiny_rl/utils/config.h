#pragma once

namespace tiny_rl
{
    struct DQNConfig
    {
        float gamma;
        float epsilon;
        float epsilon_decay;
        float epsilon_min;
        float learning_rate;
        int batch_size;
        int memory_size;
        int target_update_freq;
        int learn_start = 500;        // env steps before training begins
        int train_frequency = 4;    // how many steps between gradient updates
    };
}