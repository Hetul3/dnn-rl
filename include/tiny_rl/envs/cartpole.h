#pragma once
#include "envs/base_env.h"
#include <vector>
#include <tuple>
#include <cmath>

/*
 A sample basic cart and pole environment for learning and testing
 reasons. Environment consists of a cart that has to move left and
 right and a pole that has to be kept upright.
*/

namespace tiny_rl
{
    class CartPoleEnv : public BaseEnv
    {
    public:
        CartPoleEnv() : step_(0)
        {
            state_ = {0.0f, 0.0f, 0.05f, 0.0f}; // small pole angle to start
        }

        virtual std::vector<float> reset() override
        {
            state_ = {0.0f, 0.0f, 0.05f, 0.0f};
            step_ = 0;
            return state_;
        }

        virtual std::tuple<std::vector<float>, float, bool> step(int action) override
        {
            float delta = (action == 0 ? -0.02f : 0.02f);
            state_[2] += delta; // update pole angle
            step_++;

            bool done = (std::fabs(state_[2]) > 0.5f || step_ >= 200);
            float reward = 1.0f;

            return {state_, reward, done};
        }

        virtual int state_size() const override
        {
            return static_cast<int>(state_.size());
        }

        virtual int action_size() const override
        {
            return 2; // left or right
        }

    private:
        std::vector<float> state_;
        int step_;
    };
}