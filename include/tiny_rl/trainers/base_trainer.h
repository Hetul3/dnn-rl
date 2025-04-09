#pragma once
#include <memory>
#include <iostream>
#include "../envs/base_env.h"
#include "../agents/base_agent.h"

namespace tiny_rl
{
    class BaseTrainer
    {
    public:
        BaseTrainer(BaseAgent &agent, std::shared_ptr<BaseEnv> env)
            : agent(agent), env(env) {}

        virtual ~BaseTrainer() = default;

        virtual void train(int episodes) = 0;

    protected:
        BaseAgent &agent;
        std::shared_ptr<BaseEnv> env;
    };
}