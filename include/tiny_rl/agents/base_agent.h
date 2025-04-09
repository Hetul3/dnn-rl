#pragma once
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_rl
{
    class BaseAgent
    {
    public:
        virtual int select_action(const tiny_dnn::vec_t &state) = 0;
        
        virtual void store_experience(const tiny_dnn::vec_t &state, int action, float reward,
                                      const tiny_dnn::vec_t &next_state, bool done) = 0;

        
        virtual void learn() = 0;
      
        virtual ~BaseAgent() = default;
    };
}