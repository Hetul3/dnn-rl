#pragma once

#include <tiny_dnn/tiny_dnn.h>
#include <vector>
#include <cmath>
#include <cassert>

namespace tiny_rl
{
    class ActorCriticNetwork
    {
    public:
        ActorCriticNetwork(tiny_dnn::network<tiny_dnn::sequential> &base_net,
                           tiny_dnn::network<tiny_dnn::sequential> &policy_head,
                           tiny_dnn::network<tiny_dnn::sequential> &value_head)
            : base_net(base_net),
              policy_head(policy_head),
              value_head(value_head)
        {
        }

        std::pair<std::vector<float>, float> predict(const tiny_dnn::vec_t &state)
        {
            tiny_dnn::vec_t features = base_net.predict(state);

            tiny_dnn::vec_t logits = policy_head.predict(features);
            std::vector<float> action_probs(logits.size());
            float sum = 0.0f;
            for (size_t i = 0; i < logits.size(); ++i)
            {
                action_probs[i] = std::exp(logits[i]);
                sum += action_probs[i];
            }
            assert(sum > 0.0f);
            for (auto &p : action_probs)
                p /= sum;

            // 3) Value head → scalar
            tiny_dnn::vec_t value_vec = value_head.predict(features);
            float value = value_vec.size() > 0 ? value_vec[0] : 0.0f;

            return {action_probs, value};
        }

        // Train on one minibatch of PPO data.
        // TODO: implement PPO clipped surrogate + value + entropy losses,
        // tiny_dnn doesn't have a built-in loss function for this yet.
        void train(const std::vector<tiny_dnn::vec_t> &states,
                   const std::vector<int>              &actions,
                   const std::vector<float>            &old_log_probs,
                   const std::vector<float>            &advantages,
                   const std::vector<float>            &returns,
                   tiny_dnn::optimizer                &opt,
                   float                                clip_epsilon,
                   float                                entropy_coeff,
                   int                                  batch_size,
                   int                                  epochs = 1)
        {
        }

    private:
        tiny_dnn::network<tiny_dnn::sequential> &base_net;
        tiny_dnn::network<tiny_dnn::sequential> &policy_head;
        tiny_dnn::network<tiny_dnn::sequential> &value_head;
    };
}
