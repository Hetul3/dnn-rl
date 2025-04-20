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
              train_steps_(0)
        {
            optimizer.alpha = config.learning_rate;
            qnet.update_target_network(1.0f);
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
            bool done
        ) override
        {
            Experience exp{state, action, reward, next_state, done};
            replay_buffer.add(exp);
            ++env_steps_;
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
            if (replay_buffer.size() < config.batch_size)
                return;

            auto batch = replay_buffer.sample(config.batch_size);
            assert(batch.size() == static_cast<size_t>(config.batch_size));

            std::vector<tiny_dnn::vec_t> states, next_states;
            std::vector<int> actions;
            std::vector<float> rewards;
            std::vector<bool> dones;

            states.reserve(config.batch_size);
            next_states.reserve(config.batch_size);
            actions.reserve(config.batch_size);
            rewards.reserve(config.batch_size);
            dones.reserve(config.batch_size);

            for (auto &exp : batch)
            {
                states.push_back(exp.state);
                actions.push_back(exp.action);
                rewards.push_back(exp.reward);
                next_states.push_back(exp.next_state);
                dones.push_back(exp.done);

                assert(exp.action >= 0 && exp.action < qnet.predict(exp.state).size());
            }

            auto td_targets = qnet.compute_td_targets(
                states, actions, rewards, next_states, dones, config.gamma
            );

            qnet.train(states, td_targets, optimizer, config.batch_size);
            ++train_steps_;

            config.epsilon = std::max(
                config.epsilon_min, 
                config.epsilon * config.epsilon_decay
            );

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
    };
}