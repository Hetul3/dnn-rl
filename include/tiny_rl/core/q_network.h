#pragma once
#include <tiny_dnn/tiny_dnn.h>

namespace tiny_rl
{
    class QNetwork
    {
    public:
        QNetwork(tiny_dnn::network<tiny_dnn::sequential> &user_net)
            : net(user_net) {}

        tiny_dnn::vec_t predict(const tiny_dnn::vec_t &state)
        {
            return net.predict(state);
        }

        void train(const std::vector<tiny_dnn::vec_t> &inputs,
                   const std::vector<tiny_dnn::vec_t> &targets,
                   tiny_dnn::optimizer &opt,
                   int batch_size = 32,
                   int epochs = 1)
        {
            net.train<tiny_dnn::mse>(opt, inputs, targets, batch_size, epochs);
        }

        void train_default(const std::vector<tiny_dnn::vec_t> &inputs,
                           const std::vector<tiny_dnn::vec_t> &targets)
        {
            tiny_dnn::adagrad optimizer;
            int batch_size = 32;
            int epochs = 1;
            net.train<tiny_dnn::mse>(optimizer, inputs, targets, batch_size, epochs);
        }

        tiny_dnn::network<tiny_dnn::sequential> &get_net()
        {
            return net;
        }

    private:
        tiny_dnn::network<tiny_dnn::sequential> net;
    };
}