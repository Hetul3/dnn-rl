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
        DQNTrainer(DQNAgent &agent, std::shared_ptr<BaseEnv> env)
            : BaseTrainer(agent, env), dqn_agent(agent) {}

        void train(int episodes) override
        {
            for (int ep = 0; ep < episodes; ++ep)
            {
                auto state = env->reset();
                bool done = false;
                float steps = 0;
                float total_reward = 0.0f;

                while (!done)
                {
                    tiny_dnn::vec_t input(state.begin(), state.end());
                    int action = dqn_agent.select_action(input);

                    auto [next_state, reward, terminal] = env->step(action);
                    done = terminal;
                    total_reward += reward;

                    dqn_agent.store_experience(
                        input,
                        action,
                        reward,
                        tiny_dnn::vec_t(next_state.begin(), next_state.end()),
                        done);

                    dqn_agent.learn();
                    state = next_state;
                    ++steps;
                }
                std::cout << "Episode: " << ep + 1 << " finished in "
                          << steps << " steps, total reward: "
                          << total_reward << std::endl;
            }
        }

    private:
        DQNAgent &dqn_agent;
    };
}