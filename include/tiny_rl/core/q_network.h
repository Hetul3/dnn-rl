#pragma once
#include <tiny_dnn/tiny_dnn.h>
#include <iostream>
#include <memory>

namespace tiny_rl
{
    class QNetwork
    {

        // initialize the trainable network and the target network crucial in DQN
    public:
        QNetwork(tiny_dnn::network<tiny_dnn::sequential> &online,
                 tiny_dnn::network<tiny_dnn::sequential> &target)
            : net(online), target_net(target)
        {
            update_target_network(1.0f);
        }

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
            std::vector<tiny_dnn::vec_t> outputs;
            outputs.reserve(states.size());

            for (auto const &s : states)
            {
                outputs.push_back(use_target
                                      ? target_net.predict(s)
                                      : net.predict(s));
            }
            return outputs;
        }

        inline void clip_weights(tiny_dnn::network<tiny_dnn::sequential> &n, float limit = 6.0f)
        {
            for (size_t l = 0; l < n.depth(); ++l)
                for (auto &w : n[l]->weights())
                    for (auto &v : *w)
                    {
                        if (v > limit)
                            v = limit;
                        if (v < -limit)
                            v = -limit;
                    }
        }

        // Update network with TD targets
        void train(const std::vector<tiny_dnn::vec_t> &states, const std::vector<tiny_dnn::vec_t> &td_targets, tiny_dnn::optimizer &opt, const int batch_size = 32, const int epochs = 1)
        {
            assert(states.size() == td_targets.size());
            net.train<tiny_dnn::mse>(opt, states, td_targets, batch_size, epochs);
            clip_weights(net);
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
        int argmax_action(const tiny_dnn::vec_t &q_values)
        {
            return static_cast<int>(
                std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end())));
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
                auto next_online = predict(next_states[i], false);
                int best_act = argmax_action(next_online);
                float max_next = next_q[i][best_act];
                float td_target = rewards[i] + (dones[i] ? 0.0f : gamma * max_next);
                td_targets[i][actions[i]] = td_target;
                assert(actions[i] >= 0 &&
                       static_cast<size_t>(actions[i]) < td_targets[i].size());
            }
            return td_targets;
        }

        // Update target network weights (soft or hard update)
        void update_target_network(float tau = 1.0f)
        {
            if (tau >= 1.0f)
                tau = 1.0f; // force exact copy
            for (size_t l = 0; l < net.depth(); ++l)
            {
                auto src_params = net[l]->weights();
                auto tgt_params = target_net[l]->weights();
                for (size_t p = 0; p < src_params.size(); ++p)
                {
                    auto &src = *src_params[p];
                    auto &tgt = *tgt_params[p];
                    for (size_t i = 0; i < src.size(); ++i)
                        tgt[i] = tau * src[i] + (1.0f - tau) * tgt[i];
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
        tiny_dnn::network<tiny_dnn::sequential>& net;
        tiny_dnn::network<tiny_dnn::sequential>& target_net;
    };
}