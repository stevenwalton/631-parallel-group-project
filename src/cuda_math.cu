#include <cstdio>
#include <cstring>
#include <vector>
#include <cassert>
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "helper_cuda.h"
#include "cooperative_groups.h"

#include "cuda_math.h"

template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in)
{
  // timers
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  //// Allocate pinned memory on host (for faster HtoD copy)
  T* h_in_pinned = NULL;
  checkCudaErrors(cudaMallocHost((void**) &h_in_pinned, N * dsize));
  cudaMallocHost((void**) &h_in_pinned, N * dsize);
  assert(h_in_pinned);
  memcpy(h_in_pinned, input, N * dsize);

  // copy data
  checkCudaErrors(cudaMalloc((void**) d_in, N * dsize));
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpy(*d_in, h_in_pinned,
                             N * dsize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("  Pinned Device to Host bandwidth (GB/s): %f\n",
         (N * dsize) * 1e-6 / elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void featuresAndLabelsToGPU(std::vector<std::vector<float>>& features,
                            std::vector<std::vector<float>>& labels,
                            float** dev_features,
                            float** dev_labels)
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

    cudaMalloc((void**)dev_features, sizeof(float)*feat_size);
    //CopyData(feat_arr, feat_size, sizeof(float), dev_features);
    //float* dev_labels;
}

/*
    CopyData function taken from Jee Choi's Homework assignment
 */
