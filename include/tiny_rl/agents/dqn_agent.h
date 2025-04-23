#pragma once
#include <algorithm>
#include <iostream>
#include <random>
#include "base_agent.h"
#include "../core/q_network.h"
#include "../core/replay_buffer.h"
#include "../optim/clipped_adam.h"
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
            optimizer.b1 = 0.9f;
            optimizer.b2 = 0.999f;
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
            // don't learn if the replay buffer is not full enough for batch_size
            if (replay_buffer.size() < static_cast<size_t>(config.batch_size))
                return;

            // don't learn until we have been through minimum amount of steps
            if (env_steps_ < static_cast<size_t>(config.learn_start))
                return;

            // don't learn if the environment steps are not a multiple of the train frequency
            if (env_steps_ % config.train_frequency != 0)
                return;

            replay_buffer.sample(sample_buffer_, config.batch_size);

            float avg_done = std::accumulate(dones_.begin(), dones_.end(), 0.0f) / dones_.size();
            if (avg_done > 0.8f)
            {
                std::cout << "[BUF] done_ratio " << avg_done << std::endl;
            }

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

            if (train_steps_ % config.target_update_freq == 0 && train_steps_ > 0)
            {
                std::cout
                    << "Updating target network"
                    << std::endl;
                qnet.update_target_network(1.0f);
            }

            ++train_steps_;

            static size_t dbg_step = 0;
            if (++dbg_step % 500 == 0)
            { // print every 500 grad steps
                /* 1a. running loss (MSE) */
                float batch_loss = 0.0f;
                for (size_t i = 0; i < states_.size(); ++i)
                    batch_loss += tiny_dnn::mse::f(td_targets[i], qnet.predict(states_[i]));
                batch_loss /= states_.size();

                /* 1b. maximum |Q| in the online network */
                float q_abs_max = 0.0f;
                for (auto &q : qnet.predict_batch(states_))
                    for (float v : q)
                        q_abs_max = std::max(q_abs_max, std::fabs(v));

                /* 1c. L2-norm of all weights */
                float w_norm = 0.0f;
                for (size_t l = 0; l < qnet.get_net().depth(); ++l)
                    for (auto &W : qnet.get_net()[l]->weights())
                        for (float v : *W)
                            w_norm += v * v;
                w_norm = std::sqrt(w_norm);

                std::cout << "[DBG] step " << train_steps_
                          << "  loss " << batch_loss
                          << "  |Q|_max " << q_abs_max
                          << "  ||W||_2 " << w_norm
                          << "  eps " << config.epsilon
                          << std::endl;
            }
        }

    private:
        QNetwork &qnet;
        DQNConfig config;
        tiny_rl::clipped_adam optimizer;
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