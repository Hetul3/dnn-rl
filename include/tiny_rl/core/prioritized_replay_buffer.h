#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include "replay_buffer.h"

namespace tiny_rl
{
    class SumTree
    {
    public:
        explicit SumTree(size_t capacity)
            : capacity_(capacity),
              tree_(2 * capacity - 1, 0.0f)
        {
        }

        // overwrite leaf with new priority^alpha
        void set(size_t data_index, float priority_alpha)
        {
            size_t tree_index = data_index + capacity_ - 1;
            float delta = priority_alpha - tree_[tree_index];
            tree_[tree_index] = priority_alpha;

            // propagate up
            while (tree_index != 0)
            {
                tree_index = (tree_index - 1) / 2;
                tree_[tree_index] += delta;
            }
        }

        float total() const
        {
            return tree_[0];
        }

        size_t get_leaf(float value, float &out_priority_alpha) const
        {
            size_t index = 0;
            while (true)
            {
                size_t left = 2 * index + 1;
                if (left >= tree_.size())
                {
                    out_priority_alpha = tree_[index];
                    return index - (capacity_ - 1);
                }
                if (value <= tree_[left])
                {
                    index = left;
                }
                else
                {
                    value -= tree_[left];
                    index = left + 1;
                }
            }
        }

        void reset()
        {
            std::fill(tree_.begin(), tree_.end(), 0.0f);
        }

    private:
        size_t capacity_;
        std::vector<float> tree_;
    };

    class PrioritizedReplayBuffer
    {
    public:
        PrioritizedReplayBuffer(
            size_t capacity,
            float alpha = 0.6f,
            float beta = 0.4f)
            : tree_(capacity),
              buffer_(capacity),
              priorities_(capacity, 0.0f),
              capacity_(capacity),
              alpha_(alpha),
              beta_(beta),
              pos_(0),
              size_(0),
              rng_(std::random_device{}())
        {
        }

        // add experience with max-priority so new samples get seen at least once
        void add(const Experience &exp)
        {
            buffer_[pos_] = exp;

            float max_p = (size_ > 0) ? *std::max_element(priorities_.begin(), priorities_.begin() + size_) : 1.0f;

            priorities_[pos_] = max_p;
            tree_.set(pos_, std::pow(max_p, alpha_));

            pos_ = (pos_ + 1) % capacity_;
            if (size_ < capacity_)
                ++size_;
        }

        void sample(
            std::vector<Experience> &out,
            std::vector<size_t> &indices,
            std::vector<float> &is_weights,
            size_t batch_size)
        {
            out.clear();
            out.reserve(batch_size);
            indices.clear();
            indices.reserve(batch_size);
            is_weights.clear();
            is_weights.reserve(batch_size);

            float total_p = tree_.total();
            float segment = total_p / batch_size;
            size_t N = size_;

            // minimal sampling probaility for weight normalization
            float min_p_alpha = std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < N; ++i)
            {
                if (priorities_[i] > 0)
                {
                    float pa = std::pow(priorities_[i], alpha_);
                    min_p_alpha = std::min(min_p_alpha, pa);
                }
            }
            float min_prob = min_p_alpha / total_p;

            std::uniform_real_distribution<float> uni_dist(0.0f, 1.0f);
            for (size_t i = 0; i < batch_size; ++i)
            {
                float a = segment * i;
                float b = segment * (i + 1);
                std::uniform_real_distribution<float> dist(a, b);
                float s = dist(rng_);

                float p_alpha;
                size_t index = tree_.get_leaf(s, p_alpha);

                indices.push_back(index);
                out.push_back(buffer_[index]);

                float prob = p_alpha / total_p;
                float w = std::pow(N * prob, -beta_);
                float max_w = std::pow(N * min_prob, -beta_);
                is_weights.push_back(w / max_w);
            }
        }

        void update_priorities(
            const std::vector<size_t> &indices,
            const std::vector<float> &td_errors)
        {
            const float epsilon = 1e-6f;
            for (size_t i = 0; i < indices.size(); ++i)
            {
                size_t index = indices[i];
                float p = std::fabs(td_errors[i]) + epsilon;
                priorities_[index] = p;
                tree_.set(index, std::pow(p, alpha_));
            }
        }

        size_t size() const noexcept
        {
            return size_;
        }
        void clear() noexcept
        {
            size_ = pos_ = 0;
            std::fill(priorities_.begin(), priorities_.end(), 0.0f);
            tree_.reset();
        }

    private:
        SumTree tree_;
        std::vector<Experience> buffer_;
        std::vector<float> priorities_;
        size_t capacity_;
        float alpha_, beta_;
        size_t pos_, size_;
        std::mt19937 rng_;
    };
};