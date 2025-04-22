#pragma once
#include "base_env.h"
#include <vector>
#include <iostream>
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
            // Cartpole constants, carried from OpenAI Gym
            gravity_ = 9.8f;
            mass_cart_ = 1.0f;
            mass_pole_ = 0.1f;
            total_mass_ = mass_cart_ + mass_pole_;
            length_ = 0.5f;
            pole_mass_length_ = mass_pole_ * length_;
            force_mag_ = 10.0f;
            tau_ = 0.02f; // time (seconds) between state updates

            // Initialize state, which consists of:
            // 1. Cart position
            // 2. Cart velocity
            // 3. Pole angle (theta)
            // 4. Pole angular velocity (theta_dot)
            state_ = {0.0f, 0.0f, 0.05f, 0.0f};
        }

        const std::vector<float> get_state() const
        {
            return normalize_state(state_);
        }

        virtual std::vector<float> reset() override
        {
            state_ = {
                0.0f,
                0.0f,
                (rand() % 1000 - 500) / 10000.0f, // pole angle is slightly randomized, for the start
                0.0f};
            step_ = 0;
            return normalize_state(state_);
        }

        virtual std::tuple<std::vector<float>, float, bool> step(int action) override
        {
            float x = state_[0];
            float x_dot = state_[1];
            float theta = state_[2];
            float theta_dot = state_[3];

            float force = action == 1 ? force_mag_ : -force_mag_;
            float cos_theta = std::cos(theta);
            float sin_theta = std::sin(theta);

            float temp = (force + pole_mass_length_ * theta_dot * theta_dot * sin_theta) / total_mass_;
            float theta_acc = (gravity_ * sin_theta - cos_theta * temp) /
                              (length_ * (4.0f / 3.0f - mass_pole_ * cos_theta * cos_theta / total_mass_));
            float x_acc = temp - pole_mass_length_ * theta_acc * cos_theta / total_mass_;

            // Update state using Euler integration
            x_dot += tau_ * x_acc;
            x += tau_ * x_dot;
            theta_dot += tau_ * theta_acc;
            theta += tau_ * theta_dot;

            state_ = {x, x_dot, theta, theta_dot};
            step_++;

            // Terminal conditions: | pole angle | > 12 degrees or | cart position | > 2.4
            bool done = x < -2.4f || x > 2.4f ||
                        theta < -0.209f || theta > 0.209f ||
                        step_ >= 500;

            float reward = 1.0f;
            return {normalize_state(state_), reward, done};
        }

        virtual int state_size() const override
        {
            return static_cast<int>(state_.size());
        }

        virtual int action_size() const override
        {
            return 2; // can only move the card left or right
        }

    private:
        std::vector<float> normalize_state(const std::vector<float> &s) const
        {
            return {s[0] / 2.4f, s[1] / 3.0f, s[2] / 0.209f, s[3] / 4.0f};
        }

        std::vector<float> state_;
        int step_;
        float gravity_;
        float mass_cart_;
        float mass_pole_;
        float total_mass_;
        float length_;
        float pole_mass_length_;
        float force_mag_;
        float tau_;
    };
}