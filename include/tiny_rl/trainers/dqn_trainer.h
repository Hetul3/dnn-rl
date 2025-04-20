#pragma once

#include <iostream>
#include <tuple>
#include "base_trainer.h"
#include "../agents/dqn_agent.h"
#include "../envs/base_env.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_rl
{
    class DQNTrainer : public BaseTrainer
    {
    public:
        DQNTrainer(DQNAgent &agent,
                   std::shared_ptr<BaseEnv> env)
            : BaseTrainer(agent, env), agent_(agent) {}

        // Run episodes
        void train(int episodes) override
        {
            for (int ep = 1; ep <= episodes; ++ep)
            {
                auto raw_state = env->reset();
                tiny_dnn::vec_t state(raw_state.begin(), raw_state.end());

                float total_reward = 0.0f;
                bool done = false;

                while (!done)
                {
                    int action = agent_.select_action(state);
                    auto [next_raw_state, reward, terminal] = env->step(action);
                    tiny_dnn::vec_t next_state(next_raw_state.begin(), next_raw_state.end());

                    agent_.store_experience(state, action, reward, next_state, terminal);
                    agent_.learn();

                    state = std::move(next_state);
                    total_reward += reward;
                    done = terminal;
                }
                std::cout
                    << "Episode: "
                    << ep
                    << " finished with total reward: "
                    << total_reward
                    << std::endl;
            }
        }

    private:
        DQNAgent &agent_;
    };
}