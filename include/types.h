#ifndef TYPES_H
#define TYPES_H

#include <vector>

struct node
{
    float activation;
    std::vector<float> weight;
    std::vector<float> weight_grad;
    float error;
};

#endif
