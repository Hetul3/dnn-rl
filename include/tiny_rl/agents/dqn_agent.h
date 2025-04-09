#pragma once
#include <algorithm>
#include <iostream>
#include <random>
#include "base_agent.h"
#include "../core/q_network.h"
#include "../core/replay_buffer.h"
#include "../utils/config.h"

namespace tiny_rl
{
    class DQNAgent : public BaseAgent
    {
    public:
        DQNAgent(QNetwork &qnet, DQNConfig config)
            : qnet(qnet),
              config(config),
              replay_buffer(config.memory_size),
              rng(std::random_device{}()) {}

        int select_action(const tiny_dnn::vec_t &state) override
        {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            if (dist(rng) < config.epsilon)
            {
                int action_size = static_cast<int>(qnet.predict(state).size());
                std::uniform_int_distribution<int> action_dist(0, action_size - 1);
                return action_dist(rng);
            }
            else
            {
                auto q_values = qnet.predict(state);
                return static_cast<int>(std::distance(q_values.begin(), std::max_element(q_values.begin(), q_values.end())));
            }
        }

        void store_experience(const tiny_dnn::vec_t &state, int action, float reward,
                              const tiny_dnn::vec_t &next_state, bool done) override
        {
            Experience exp{state, action, reward, next_state, done};
            replay_buffer.add(exp);
        }

        void learn() override
        {
            if (replay_buffer.size() < static_cast<size_t>(config.batch_size))
            {
                return;
            }

            auto batch = replay_buffer.sample(config.batch_size);
            std::vector<tiny_dnn::vec_t> inputs;
            std::vector<tiny_dnn::vec_t> targets;

            for (auto &exp : batch)
            {
                auto current_q = qnet.predict(exp.state);
                auto next_q = qnet.predict(exp.next_state);
                float max_next_q = *std::max_element(next_q.begin(), next_q.end());
                float target_q = exp.done ? exp.reward : exp.reward + config.gamma * max_next_q;
                current_q[exp.action] = target_q;
                inputs.push_back(exp.state);
                targets.push_back(current_q);
            }

            qnet.train_default(inputs, targets);

            config.epsilon = std::max(config.epsilon_min, config.epsilon * config.epsilon_decay);
        }

    private:
        QNetwork &qnet;
        DQNConfig config;
        ReplayBuffer replay_buffer;
        std::mt19937 rng;
    };
}