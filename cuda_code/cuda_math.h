#ifndef CUDA_MATH_H
#define CUDA_MATH_H 1

#include "common.h"

template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in);

void featuresAndLabelsToGPU(std::vector<std::vector<float>>&,
                            std::vector<int>&,
                            size_t,
                            float**,
                            int**,
                            float**);

template <class T>
void matrix2Cuda(std::vector<std::vector<T>>& m, T** dev_m);

template <class T>
void vector2Cuda(std::vector<T>& v, T** dev_v);

void cudaMatrixMultiply(std::vector<std::vector<float>>,
                        std::vector<std::vector<float>>,
                        std::vector<std::vector<float>>&);

void matrixToCuda(std::vector<std::vector<float>>&,
                  std::vector<std::vector<float>>&,
                  float**,
                  float**,
                  float**);

void cudaToMatrix(float**,
                  std::vector<std::vector<float>>&);
//void modelToGPU(Sequential&);


#endif
