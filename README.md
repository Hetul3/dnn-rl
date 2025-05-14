# dnn-rl

A header-only C++ reinforcement-learning extension built on top of [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn).
`dnn-rl` provides ready-to-use components for developing and evaluating RL agents in classic control environments.

---

## Features

* **Header-only** — just include `include/` and `external/tiny-dnn/`
* **Algorithms**

  * Deep Q-Network (DQN)
  * Proximal Policy Optimization (PPO)
* **Environments**

  * `CartPoleEnv` (implemented)
  * `GridWorldEnv` (coming soon)
* **Config structs** for easy hyperparameter tuning

  * `tiny_rl::DQNConfig`
  * `tiny_rl::PPOConfig`
* **Unit tests** covering environments, agents, and trainers
* **Documentation** under `docs/`

---

## Repository layout

```
.
├── include/                   # public headers
│   └── tiny_rl/
│       ├── core/              # replay buffer, Q-network
│       ├── agents/            # base_agent, dqn_agent
│       ├── trainers/          # base_trainer, dqn_trainer
│       ├── envs/              # cartpole.h, gridworld.h
│       └── utils/             # config.h (DQNConfig, PPOConfig)
├── external/
│   └── tiny-dnn/              # header-only tiny-dnn library
├── tests/
│   ├── CMakeLists.txt
│   ├── Makefile               # `make`, `make runtests:<test_name>`
│   ├── test_envs.cpp
│   ├── test_agents.cpp
│   └── test_trainers.cpp
├── docs/
│   ├── overview.md
│   ├── architecture.md
│   └── agents_api.md
├── examples/                  # sample programs (coming soon)
└── README.md
```

---

## Installation & Integration

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/dnn-rl.git
   cd dnn-rl
   ```

2. **Include paths**

   * `-Iinclude`
   * `-Iexternal/tiny-dnn`

3. **Compiler flags**

   * C++17 or later (e.g. `-std=c++17`)
   * No linker flags required (header-only)

---

## Usage

*Add the following to your build command or CMakeLists:*

```cmake
# If you already build tiny-dnn as a subproject:
add_subdirectory(external/tiny-dnn)

add_library(tiny_rl INTERFACE)
target_include_directories(tiny_rl INTERFACE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(tiny_rl INTERFACE tiny_dnn)
```

*In a simple GCC/Clang command-line build:*

```bash
g++ -std=c++17 \
    -Iinclude \
    -Iexternal/tiny-dnn \
    your_app.cpp \
    -o your_app
```

*In your code:*

```cpp
#include "tiny_rl.h"
#include <tiny_dnn/tiny_dnn.h>

int main() {
    auto env = std::make_shared<tiny_rl::CartPoleEnv>();

    tiny_dnn::network<tiny_dnn::sequential> online, target;
    auto build_net = [&](auto &n) {
        n << tiny_dnn::fully_connected_layer(env->state_size(), 64)
          << tiny_dnn::relu_layer()
          << tiny_dnn::fully_connected_layer(64, env->action_size());
    };
    build_net(online);
    build_net(target);

    tiny_rl::QNetwork qnet(online, target);

    tiny_rl::DQNAgent agent(qnet, tiny_rl::DQNConfig{
        .gamma            = 0.99f,
        .epsilon          = 1.0f,
        .epsilon_decay    = 0.9995f,
        .epsilon_min      = 0.05f,
        .learning_rate    = 0.001f,
        .batch_size       = 64,
        .memory_size      = 5000,
        .target_update_freq = 2000
    });
    tiny_rl::DQNTrainer trainer(agent, env);

    trainer.train(10000);
    return 0;
}
```

---

## Running the tests

From the `tests/` directory:

```bash
# Build & run all tests
make
make runtests:test_envs
make runtests:test_agents
make runtests:test_trainers

# Or run a specific test:
make runtests:test_envs
```

*Or with CMake:*

```bash
cd tests
mkdir build && cd build
cmake ..
make
```

---

## Configuration structs

```cpp
// DQN hyperparameters
struct DQNConfig {
    float gamma;
    float epsilon;
    float epsilon_decay;
    float epsilon_min;
    float learning_rate;
    int   batch_size;
    int   memory_size;
    int   target_update_freq;
    int   learn_start     = 500;  // steps before training begins
    int   train_frequency =   4;  // steps between gradient updates
};

// PPO hyperparameters
struct PPOConfig {
    float gamma         = 0.99f;
    float lambda        = 0.95f;
    float clip_epsilon  = 0.2f;
    float learning_rate = 3e-4f;
    float entropy_coeff = 0.01f;
    int   batch_size    = 64;
    int   mini_epochs   = 4;
    int   buffer_capacity = 2048;
};
```

---

## Roadmap

* CartPole environment
* DQN & PPO agents
* GridWorldEnv (coming soon)
* Populate `examples/` with training scripts
