#pragma once
#include <vector>
#include <tuple>

namespace tiny_rl
{
    class BaseEnv
    {
    public:
        // Resets the environment and returns the intiial state
        virtual std::vector<float> reset() = 0;

        // Takes an action and returns the next state, reward, and done flag
        virtual std::tuple<std::vector<float>, float, bool> step(int action) = 0;

        // Retuns the size of the state vector
        virtual int state_size() const = 0;

        // Returns the number of possible actions
        virtual int action_size() const = 0;

        virtual ~BaseEnv() = default;
    };

}