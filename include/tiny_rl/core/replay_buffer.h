#pragma once
#include <vector>
#include <cstdlib>
#include <random>
#include <algorithm>
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_rl
{
    struct Experience
    {
        tiny_dnn::vec_t state;
        int action;
        float reward;
        tiny_dnn::vec_t next_state;
        bool done;
    };

    class ReplayBuffer
    {
    public:
        explicit ReplayBuffer(size_t capacity)
            : capacity(capacity), index(0)
        {
            buffer.reserve(capacity);
        }

        void add(const Experience &exp)
        {
            if (buffer.size() < capacity)
            {
                buffer.push_back(exp);
            }
            else
            {
                buffer[index] = exp;
            }
            index = (index + 1) % capacity;
        }

        // Randomly sample a batch of experiences from the buffer
        std::vector<Experience> sample(size_t batch_size)
        {
            if (buffer.empty())
            {
                return {};
            }

            std::vector<Experience> batch;
            batch.reserve(batch_size);

            std::uniform_int_distribution<size_t> dist(0, buffer.size() - 1);
            for (size_t i = 0; i < batch_size; ++i)
            {
                batch.push_back(buffer[dist(rng)]);
            }
            return batch;
        }

        size_t size() const
        {
            return buffer.size();
        }

        void clear()
        {
            buffer.clear();
            index = 0;
        }

    private:
        size_t capacity;
        size_t index;
        std::vector<Experience> buffer;
        std::mt19937 rng{std::random_device{}()};
    };
}