#ifndef TYPES_H
#define TYPES_H

#include <vector>

struct node
{
    float activation;
    float bias;
    std::vector<float> weight;
    //std::vector<float> weight_grad;
    float delta; //gradient
    float error;
};

#endif
