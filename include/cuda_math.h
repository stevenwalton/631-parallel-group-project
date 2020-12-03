#ifndef CUDA_MATH_H
#define CUDA_MATH_H 1

#include "common.h"

template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in);

void featuresAndLabelsToGPU(std::vector<std::vector<float>>& features,
                            std::vector<std::vector<float>>& labels,
                            float** dev_features,
                            float** dev_labels);

#endif
