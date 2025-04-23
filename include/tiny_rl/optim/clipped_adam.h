#pragma once
#include "../external/tiny-dnn/tiny_dnn/optimizers/optimizer.h"
#include <cmath>

namespace tiny_rl
{
    struct clipped_adam : public tiny_dnn::adam
    {
        explicit clipped_adam(float_t max_norm = float_t(5.0))
            : max_norm_(max_norm) {}

        void update(const tiny_dnn::vec_t &dW,
                    tiny_dnn::vec_t &W,
                    bool parallelize) override
        {

            // L2-norm of gradient
            float_t sq = float_t(0);
            for (float_t g : dW)
                sq += g * g;
            float_t norm = std::sqrt(sq);

            if (norm > max_norm_)
            {
                float_t scale = max_norm_ / (norm + 1e-8f);
                tiny_dnn::vec_t clipped(dW.size());
                for (size_t i = 0; i < dW.size(); ++i)
                    clipped[i] = dW[i] * scale;

                tiny_dnn::adam::update(clipped, W, parallelize);
            }
            else
            {
                tiny_dnn::adam::update(dW, W, parallelize);
            }
        }

        float_t max_norm_;
    };

}
