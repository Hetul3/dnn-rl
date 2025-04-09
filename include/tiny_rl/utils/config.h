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
    };
}