#pragma once
#include <tiny_dnn/tiny_dnn.h>
#include <memory>

namespace tiny_rl
{
    class QNetwork
    {

        // initialize the trainable network and the target network crucial in DQN
    public:
        QNetwork(tiny_dnn::network<tiny_dnn::sequential> &user_net)
            : net(user_net), target_net(user_net) {}

        // get Q-values for a single state
        // Takes in the env state and returns the Q-values for the actions
        tiny_dnn::vec_t predict(const tiny_dnn::vec_t &state, bool use_target = false)
        {
            if (use_target)
            {
                return target_net.predict(state);
            }
            return net.predict(state);
        }

        // get Q-values for a batch of states
        std::vector<tiny_dnn::vec_t> predict_batch(const std::vector<tiny_dnn::vec_t> &states, bool use_target = false)
        {
            if (use_target)
            {
                return target_net.predict(states);
            }
            return net.predict(states);
        }

        // Update network with TD targets
        void train(const std::vector<tiny_dnn::vec_t> &states, const std::vector<tiny_dnn::vec_t> &td_targets, tiny_dnn::optimizer &opt, const int batch_size = 32, const int epochs = 1)
        {
            assert(states.size() == td_targets.size());
            net.train<tiny_dnn::mse>(opt, states, td_targets, batch_size, epochs);
        }

        // Get max Q-value for a state
        float get_max_q_value(const tiny_dnn::vec_t &q_values)
        {
            float max = q_values[0];
            for (size_t i = 1; i < q_values.size(); ++i)
            {
                if (q_values[i] > max)
                {
                    max = q_values[i];
                }
            }
            return max;
        }

        // Get max Q-value index for a state, helpful for the agent
        int argmax_action(const tiny_dnn::vec_t &q_values) {
            return static_cast<int>(
                std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end()))
            );
        }

        // Compute TD targets: r + gamma * max_a' Q(s', a')
        std::vector<tiny_dnn::vec_t> compute_td_targets(
            const std::vector<tiny_dnn::vec_t> &states,
            const std::vector<int> &actions,
            const std::vector<float> &rewards,
            const std::vector<tiny_dnn::vec_t> &next_states,
            const std::vector<bool> &dones,
            float gamma = 0.99f)
        {
            size_t N = states.size();
            assert(actions.size() == N);
            assert(rewards.size() == N);
            assert(next_states.size() == N);
            assert(dones.size() == N);

            auto current_q = predict_batch(states, false);
            auto next_q = predict_batch(next_states, true);

            std::vector<tiny_dnn::vec_t> td_targets = current_q;
            for (size_t i = 0; i < rewards.size(); ++i)
            {
                float max_next = get_max_q_value(next_q[i]);
                float td_target = rewards[i] + (dones[i] ? 0.0f : gamma * max_next);
                td_targets[i][actions[i]] = td_target;
                assert(actions[i] >= 0 && actions[i] < td_targets[i].size());
            }
            return td_targets;
        }

        // Update target network weights (soft or hard update)
        void update_target_network(float tau = 1.0f)
        {
            if (tau >= 1.0f)
            {
                target_net = net;
            }
            else
            {
                // soft update: θ_target ← τ·θ_net + (1–τ)·θ_target
                for (size_t l = 0; l < net.depth(); ++l)
                {
                    auto src_layer = net[l];
                    auto tgt_layer = target_net[l];

                    auto src_params = src_layer->weights();
                    auto tgt_params = tgt_layer->weights();

                    for (size_t p = 0; p < src_params.size(); ++p)
                    {
                        auto &src = *src_params[p];
                        auto &tgt = *tgt_params[p];

                        for (size_t i = 0; i < src.size(); ++i)
                        {
                            tgt[i] = tau * src[i] + (1.0f - tau) * tgt[i];
                        }
                    }
                }
            }
        }

        // getters
        tiny_dnn::network<tiny_dnn::sequential> &get_net()
        {
            return net;
        }

        tiny_dnn::network<tiny_dnn::sequential> &get_target()
        {
            return target_net;
        }

    private:
        tiny_dnn::network<tiny_dnn::sequential> net;
        tiny_dnn::network<tiny_dnn::sequential> target_net;
    };
}