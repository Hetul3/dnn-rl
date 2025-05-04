#pragma once

#include <vector>
#include <stdexcept>
#include <cstddef>
#include <tiny_dnn/tiny_dnn.h>

namespace tiny_rl
{
    struct RolloutEntry
    {
        tiny_dnn::vec_t state;
        int action;
        float reward;
        bool done;
        float log_prob;
        float value;
        float advantage = 0.0f;
        float return_ = 0.0f;
    };

    class RolloutBuffer
    {
    public:
        RolloutBuffer(size_t capacity)
            : capacity_(capacity)
        {
            buffer_.reserve(capacity);
        }

        void add(const RolloutEntry &entry)
        {
            if (buffer_.size() < capacity_)
            {
                buffer_.push_back(entry);
            }
            else
            {
                throw std::runtime_error("RolloutBuffer is full");
            }
        }

        void clear()
        {
            buffer_.clear();
        }

        size_t size() const
        {
            return buffer_.size();
        }

        bool full() const
        {
            return buffer_.size() >= capacity_;
        }

        const std::vector<RolloutEntry> &data() const
        {
            return buffer_;
        }

        std::vector<RolloutEntry> &mutable_data()
        {
            return buffer_;
        }

    private:
        std::vector<RolloutEntry> buffer_;
        size_t capacity_;
    };
}