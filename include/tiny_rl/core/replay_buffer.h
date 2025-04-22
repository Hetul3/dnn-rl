#pragma once
#include <vector>
#include <cstdlib>
#include <random>
#include <cassert>
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
            : capacity(capacity),
              size_(0),
              pos_(0),
              buffer_(capacity),
              rng_(std::random_device{}()),
              dist_(0, capacity - 1)
        {
        }

        void add(Experience &&exp)
        {
            buffer_[pos_] = std::move(exp);
            advance_();
        }

        void add(const Experience &exp)
        {
            buffer_[pos_] = exp;
            advance_();
        }

        // Randomly sample a batch of experiences from the buffer
        void sample(std::vector<Experience> &out, size_t batch_size)
        {
            assert(size_ > 0);
            assert(batch_size <= size_);

            out.clear();
            out.resize(batch_size);
            dist_.param(typename decltype(dist_)::param_type(0, size_ - 1));

            for (size_t i = 0; i < batch_size; ++i)
            {
                out[i] = buffer_[dist_(rng_)];
            }
        }

        size_t size() const noexcept
        {
            return size_;
        }

        void clear() noexcept
        {
            size_ = pos_ = 0;
        }

    private:
        void advance_()
        {
            pos_ = (pos_ + 1) % capacity;
            if (size_ < capacity)
            {
                ++size_;
            }
        }

        size_t capacity;
        size_t size_;
        size_t pos_;
        std::vector<Experience> buffer_;
        std::mt19937 rng_;
        std::uniform_int_distribution<size_t> dist_;
    };
}