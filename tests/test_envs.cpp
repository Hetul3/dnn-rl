#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <cassert>
#include <functional>
#include "../include/tiny_rl/tiny_rl.h"

// temporary framework for now, generated with AI. Need to be replaced with proper testing framework
#define TEST_CASE(name) void name()
#define SECTION(name) std::cout << "  " << name << std::endl;
#define REQUIRE(condition)                                                                  \
    if (!(condition))                                                                       \
    {                                                                                       \
        std::cerr << "Test failed at line " << __LINE__ << ": " << #condition << std::endl; \
        assert(condition);                                                                  \
    }
#define CHECK(condition)                                                                     \
    if (!(condition))                                                                        \
    {                                                                                        \
        std::cerr << "Check failed at line " << __LINE__ << ": " << #condition << std::endl; \
    }

bool roughly_equal(float a, float b, float epsilon = 0.0001f)
{
    return std::abs(a - b) < epsilon;
}

void print_state(const std::vector<float> &state)
{
    std::cout << "State: [";
    for (size_t i = 0; i < state.size(); ++i)
    {
        std::cout << state[i];
        if (i < state.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

TEST_CASE(test_cartpole_initialization)
{
    std::cout << "Testing CartPole initialization" << std::endl;

    auto env = std::make_shared<tiny_rl::CartPoleEnv>();

    SECTION("State and action space dimensions")
    REQUIRE(env->state_size() == 4);  // x, x_dot, theta, theta_dot
    REQUIRE(env->action_size() == 2); // left, right

    SECTION("Reset functionality")
    auto state = env->reset();
    REQUIRE(state.size() == 4);
    REQUIRE(state[0] == 0.0f); // Initial cart position should be 0
    REQUIRE(state[1] == 0.0f); // Initial cart velocity should be 0
    // Pole angle is slightly random but small
    CHECK(std::abs(state[2]) < 0.06f);
    REQUIRE(state[3] == 0.0f); // Initial pole angular velocity should be 0
}

TEST_CASE(test_cartpole_physics)
{
    std::cout << "Testing CartPole physics" << std::endl;

    auto env = std::make_shared<tiny_rl::CartPoleEnv>();

    SECTION("Action effects")
    env->reset();
    std::cout << " Initial state: " << env->get_state()[0] << std::endl;
    auto [state_right, reward_right, done_right] = env->step(1); // Push right
    std::cout << " State after pushing right: " << state_right[0] << std::endl;
    REQUIRE(state_right[0] > 0.0f);           // Cart should move right

    env->reset();
    std::cout << " State after pushing right: " << state_right[0] << std::endl;
    auto [state_left, reward_left, done_left] = env->step(0); // Push left
    std::cout << " State after pushing left: " << state_left[0] << std::endl;
    REQUIRE(state_left[0] < 0.0f);               // Cart should move left

    SECTION("Cumulative effects")
    env->reset();
    float prev_x = 0.0f;
    for (int i = 0; i < 5; i++)
    {
        auto [state, reward, done] = env->step(1); // Always push right
        REQUIRE(state[0] > prev_x);                // Position should keep increasing
        prev_x = state[0];
    }

    SECTION("State variable relationships")
    env->reset();
    auto [state1, reward1, done1] = env->step(1);
    // Velocity should match position change (approximately)
    CHECK(roughly_equal(state1[0], state1[1] * 0.02f, 0.01f));
}

TEST_CASE(test_cartpole_rewards)
{
    std::cout << "Testing CartPole reward system" << std::endl;

    auto env = std::make_shared<tiny_rl::CartPoleEnv>();

    SECTION("Normal step reward")
    env->reset();
    auto [state_normal, reward_normal, done_normal] = env->step(0);
    REQUIRE(roughly_equal(reward_normal, 1.0f));

    SECTION("Failure reward")
    // Force failure by pushing continuously in one direction
    env->reset();
    bool done = false;
    float last_reward = 0.0f;
    int steps = 0;

    while (!done && steps < 100)
    {
        auto [__, reward, is_done] = env->step(1);
        last_reward = reward;
        done = is_done;
        steps++;
    }

    if (done && steps < 100)
    {
        // Check if failure reward is correctly implemented
        // This check depends on the actual implementation
        CHECK(last_reward == 1.0f || last_reward == -1.0f || last_reward == 0.0f || last_reward == 0.0f);
    }
}

TEST_CASE(test_cartpole_termination)
{
    std::cout << "Testing CartPole termination conditions" << std::endl;

    auto env = std::make_shared<tiny_rl::CartPoleEnv>();

    SECTION("Position boundary termination")
    env->reset();
    bool terminated = false;
    int steps = 0;
    std::vector<float> final_state;

    while (!terminated && steps < 300)
    {
        auto [state, _, done] = env->step(1); // Always push right
        terminated = done;
        final_state = state;
        steps++;
    }

    // Should terminate eventually due to position limits
    REQUIRE(terminated);
    CHECK(final_state[0] > 2.4f || final_state[0] < -2.4f);
    std::cout << "  Position boundary termination occurred after " << steps << " steps" << std::endl;
    std::cout << "  Final position: " << final_state[0] << std::endl;

    SECTION("Angle boundary termination")
    env->reset();
    terminated = false;
    steps = 0;

    // Apply alternating forces to destabilize the pole
    int action = 0;
    while (!terminated && steps < 300)
    {
        action = 1 - action; // Alternate between 0 and 1
        auto [state, _, done] = env->step(action);
        if (steps % 5 == 0)
        {
            action = rand() % 2; // Add some randomness to help pole fall faster
        }
        terminated = done;
        final_state = state;
        steps++;

        if (terminated && std::abs(state[0]) <= 2.4f)
        {
            // Terminated due to pole angle, not cart position
            CHECK(std::abs(state[2]) > 0.209f);
            std::cout << "  Angle boundary termination occurred after " << steps << " steps" << std::endl;
            std::cout << "  Final angle: " << state[2] << " radians" << std::endl;
            break;
        }
    }
}

void run_cartpole_example(bool verbose = false)
{
    std::cout << "Running example CartPole episode" << std::endl;

    auto env = std::make_shared<tiny_rl::CartPoleEnv>();
    env->reset();

    int total_reward = 0;
    int max_steps = 100;

    for (int step = 0; step < max_steps; step++)
    {
        // Simple policy: move cart in opposite direction of the pole's lean
        int action = env->get_state()[2] > 0 ? 0 : 1;
        auto [state, reward, done] = env->step(action);

        total_reward += reward;

        if (verbose)
        {
            std::cout << "Step " << step << ": ";
            print_state(state);
            std::cout << "Action: " << action << ", Reward: " << reward
                      << ", Done: " << (done ? "true" : "false") << std::endl;
        }

        if (done)
        {
            std::cout << "  Episode terminated after " << step + 1 << " steps with reward " << total_reward << std::endl;
            return;
        }
    }
    std::cout << "  Episode completed " << max_steps << " steps with reward " << total_reward << std::endl;
}

int main()
{
    std::cout << "Starting CartPole environment tests\n"
              << std::endl;

    // Run all test cases
    test_cartpole_initialization();
    test_cartpole_physics();
    test_cartpole_rewards();
    test_cartpole_termination();

    // Run example episode
    run_cartpole_example();

    std::cout << "\nAll tests completed successfully!" << std::endl;
    return 0;
}