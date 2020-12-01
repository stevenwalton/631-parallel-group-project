#include <cstdio>
#include <vector>
#include <cassert>
#include "helper_cuda.h"
#include "cooperative_groups.h"

void featuresAndLabelsToGPU(std::vector<std::vector<float>>& features,
                            std::vector<std::vector<float>>& labels)
{
    // Flatten vector of vectors and push to arrays
    size_t f_size = features.size();
    size_t f0_size = features[0].size();
    size_t feat_size = f_size * f0_size;
    float* feat_arr = (float*)malloc(feat_size * sizeof(float));
    for (size_t i = 0; i < f_size; ++i)
        for (size_t j = 0; j < f0_size; ++j)
            feat_arr[i*f0_size + j] = features[i][j];
    assert(feat_arr);

    size_t l_size = labels.size();
    size_t l0_size = labels[0].size();
    size_t label_size = l_size * l0_size;
    float* label_arr = (float*)malloc(label_size * sizeof(float));
    for (size_t i = 0; i < l_size; ++i)
        for(size_t j = 0; j < l0_size; ++j)
            label_arr[i*l0_size + j] = labels[i][j];
    assert(label_arr);
}
