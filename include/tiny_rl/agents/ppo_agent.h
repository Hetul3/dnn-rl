// tiny_rl/agents/ppo_agent.h
#pragma once

#include <algorithm>
#include <iostream>
#include <random>

#include "base_agent.h"
#include "../core/actor_critic_network.h"
#include "../core/rollout_buffer.h"
#include "../optim/clipped_adam.h"
#include "../utils/config.h"

namespace tiny_rl
{
    class PPOAgent : public BaseAgent
    {
    public:
        PPOAgent(ActorCriticNetwork &ac_net, PPOConfig config)
            : ac_net(ac_net),
              config(config),
              rollout_buffer(config.buffer_capacity),
              rng(std::random_device{}()),
              env_steps_(0),
              train_steps_(0)
        {
            optimizer.alpha = config.learning_rate;
            optimizer.b1 = 0.9f;
            optimizer.b2 = 0.999f;
        }

        int select_action(const tiny_dnn::vec_t &state) override
        {
            auto [probs, value] = ac_net.predict(state);
            std::discrete_distribution<int> dist(probs.begin(), probs.end());
            int action = dist(rng);

            last_log_prob_ = std::log(probs[action] + 1e-8f);
            last_value_ = value;
            return action;
        }

        void store_experience(const tiny_dnn::vec_t &state,
                              int action,
                              float reward,
                              const tiny_dnn::vec_t &,
                              bool done) override
        {
            RolloutEntry entry;
            entry.state = state;
            entry.action = action;
            entry.reward = reward;
            entry.done = done;
            entry.log_prob = last_log_prob_;
            entry.value = last_value_;

            rollout_buffer.add(entry);
            ++env_steps_;
        }

        void learn() override
        {
            // only train once buffer is full
            if (!rollout_buffer.full())
                return;

            compute_gae_and_returns();

            const auto &data = rollout_buffer.data();
            size_t N = data.size();
            std::vector<tiny_dnn::vec_t> states;
            states.reserve(N);
            std::vector<int> actions;
            actions.reserve(N);
            std::vector<float> old_log_probs;
            old_log_probs.reserve(N);
            std::vector<float> advantages;
            advantages.reserve(N);
            std::vector<float> returns;
            returns.reserve(N);

            for (const auto &e : data)
            {
                states.push_back(e.state);
                actions.push_back(e.action);
                old_log_probs.push_back(e.log_prob);
                advantages.push_back(e.advantage);
                returns.push_back(e.return_);
            }

            ac_net.train(states,
                         actions,
                         old_log_probs,
                         advantages,
                         returns,
                         optimizer,
                         config.clip_epsilon,
                         config.entropy_coeff,
                         config.batch_size,
                         config.mini_epochs);

            rollout_buffer.clear();
            ++train_steps_;
            if (train_steps_ % 500 == 0)
                std::cout << "[PPOAgent] Completed update #" << train_steps_ << std::endl;
        }

        void on_episode_end() override
        {
        }

        void reset() override
        {
            rollout_buffer.clear();
        }

        void seed(unsigned int seed) override
        {
            rng.seed(seed);
        }

    private:
        void compute_gae_and_returns()
        {
            auto &data = rollout_buffer.mutable_data();
            float gae = 0.0f;
            float next_value = data.back().value;

            for (int t = static_cast<int>(data.size()) - 1; t >= 0; --t)
            {
                float delta = data[t].reward + (data[t].done ? 0.0f : config.gamma * next_value) - data[t].value;
                gae = delta + config.gamma * config.lambda * (data[t].done ? 0.0f : gae);
                data[t].advantage = gae;
                data[t].return_ = gae + data[t].value;
                next_value = data[t].value;
            }
        }

        ActorCriticNetwork &ac_net;
        PPOConfig config;
        RolloutBuffer rollout_buffer;
        tiny_rl::clipped_adam optimizer;
        std::mt19937 rng;
        size_t env_steps_;
        size_t train_steps_;
        float last_log_prob_;
        float last_value_;
    };
}
