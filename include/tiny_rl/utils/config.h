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

    struct PPOConfig
    {
        float gamma = 0.99f;
        float lambda = 0.95f;
        float clip_epsilon = 0.2f;
        float learning_rate = 3e-4f;
        float entropy_coeff = 0.01f;
        int batch_size = 64;
        int mini_epochs = 4;
        int buffer_capacity = 2048;
    };
}