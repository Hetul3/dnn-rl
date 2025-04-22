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
              rng(std::random_device{}()),
              env_steps_(0),
              train_steps_(0),
              sample_buffer_()
        {
            optimizer.alpha = config.learning_rate;
            qnet.update_target_network(1.0f);
            sample_buffer_.reserve(config.batch_size);
        }

        // Select action based on epsilon-greedy policy
        // Select a weighted random action with probability epsilon
        int select_action(const tiny_dnn::vec_t &state) override
        {
            std::uniform_real_distribution<float> coin(0, 1);
            if (coin(rng) < config.epsilon)
            {
                int num_actions = qnet.predict(state).size();
                std::uniform_int_distribution<int> pick(0, num_actions - 1);
                return pick(rng);
            }
            auto q_values = qnet.predict(state);
            return qnet.argmax_action(q_values);
        }

        // Store the experience in the replay buffer
        void store_experience(
            const tiny_dnn::vec_t &state,
            int action, float reward,
            const tiny_dnn::vec_t &next_state,
            bool done) override
        {
            Experience exp{state, action, reward, next_state, done};
            replay_buffer.add(exp);
            ++env_steps_;
        }

        void on_episode_end() override
        {
            config.epsilon = std::max(config.epsilon_min,
                                      config.epsilon * config.epsilon_decay);
        }

        void learn() override
        {
            // don't learn until we have been through minimum amount of steps
            if (env_steps_ < static_cast<size_t>(config.learn_start))
                return;

            // don't learn if the environment steps are not a multiple of the train frequency
            if (env_steps_ % config.train_frequency != 0)
                return;

            // don't learn if the replay buffer is not full yet
            if (replay_buffer.size() < static_cast<size_t>(config.batch_size))
                return;

            replay_buffer.sample(sample_buffer_, config.batch_size);

            states_.clear();
            next_states_.clear();
            actions_.clear();
            rewards_.clear();
            dones_.clear();

            for (const auto &exp : sample_buffer_)
            {
                states_.push_back(exp.state);
                actions_.push_back(exp.action);
                rewards_.push_back(exp.reward);
                next_states_.push_back(exp.next_state);
                dones_.push_back(exp.done);
            }

            auto td_targets = qnet.compute_td_targets(
                states_, actions_, rewards_, next_states_, dones_, config.gamma);

            qnet.train(states_, td_targets, optimizer, config.batch_size);
            ++train_steps_;

            if (train_steps_ % config.target_update_freq == 0)
            {
                qnet.update_target_network(1.0f);
            }
        }

    private:
        QNetwork &qnet;
        DQNConfig config;
        tiny_dnn::adam optimizer;
        ReplayBuffer replay_buffer;
        std::mt19937 rng;
        size_t env_steps_;
        size_t train_steps_;

        std::vector<Experience> sample_buffer_;
        std::vector<tiny_dnn::vec_t> states_;
        std::vector<tiny_dnn::vec_t> next_states_;
        std::vector<int> actions_;
        std::vector<float> rewards_;
        std::vector<bool> dones_;
    };
}