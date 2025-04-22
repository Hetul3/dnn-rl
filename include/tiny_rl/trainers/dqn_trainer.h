#pragma once

#include <iostream>
#include <tuple>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
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
            : BaseTrainer(agent, env), agent_(agent), paused_(false) {}

        // Run episodes
        void train(int episodes) override
        {
            float avg_reward_100 = 0.0f;

            // Start the input monitoring thread
            std::atomic<bool> stop_input_thread(false);
            std::thread input_thread([this, &stop_input_thread]()
                                     { this->monitor_input(stop_input_thread); });

            for (int ep = 1; ep <= episodes; ++ep)
            {
                // Check if paused before starting a new episode
                check_pause_status();

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
                agent_.on_episode_end();
                avg_reward_100 += total_reward;

                if (ep % 100 == 0)
                {
                    std::cout << "Episode: " << ep
                              << " average reward: " << avg_reward_100 / 100
                              << std::endl;

                    avg_reward_100 = 0.0f;
                }
            }

            // Stop and join the input thread when training is complete
            stop_input_thread = true;
            if (input_thread.joinable())
            {
                input_thread.join();
            }
        }

    private:
        DQNAgent &agent_;
        std::atomic<bool> paused_;
        std::mutex pause_mutex_;
        std::condition_variable pause_cv_;

        // Monitor for pause/resume commands
        void monitor_input(std::atomic<bool> &stop_flag)
        {
            std::string input;
            while (!stop_flag)
            {
                std::getline(std::cin, input);
                if (input == "pause")
                {
                    std::cout << "Training will pause after current episode completes..." << std::endl;
                    paused_ = true;
                }
                else if (input == "resume")
                {
                    std::cout << "Resuming training..." << std::endl;
                    {
                        std::lock_guard<std::mutex> lock(pause_mutex_);
                        paused_ = false;
                    }
                    pause_cv_.notify_one();
                }
            }
        }

        // Check if training should be paused
        void check_pause_status()
        {
            if (paused_)
            {
                std::cout << "Training paused. Type 'resume' + Enter to continue." << std::endl;
                std::unique_lock<std::mutex> lock(pause_mutex_);
                pause_cv_.wait(lock, [this]
                               { return !paused_; });
            }
        }
    };
}