#include <cstdio>
#include <vector>
#include <cassert>
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "helper_cuda.h"
#include "cooperative_groups.h"

#include "cuda_math.h"

// Kernels go up top
template <class T>
__global__ void
cuda_matrix_mult(size_t x_size, size_t y_size,
                 T* x, T* y, T* z)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    if ( row < x_size && col < y_size)
    {
        T tmp = 0;
        for (size_t k = 0; k < y_size; ++k)
            tmp += x[row*y_size+ k] * y[k*y_size + col];
        z[row*y_size+ col] = tmp;
    }
}



/*
    CopyData function taken from Jee Choi's Homework assignment
 */
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

  // Allocate pinned memory on host (for faster HtoD copy)
  T* h_in_pinned = NULL;
  checkCudaErrors(cudaMallocHost((void**) &h_in_pinned, N * dsize));
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
                            size_t batch_size,
                            float** dev_features,
                            float** dev_labels,
                            float** dev_predictions)
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

    CopyData(feat_arr, feat_size, sizeof(float), dev_features);
    CopyData(label_arr, label_size, sizeof(float), dev_labels);

    // TODO
    // Check that preds is the correct shape and size
    float* preds = (float*)malloc(batch_size * sizeof(float));
    memset(preds, 0, batch_size * sizeof(float));
    CopyData(preds, batch_size, sizeof(float), dev_predictions);
}

void matrixToCuda(std::vector<std::vector<float>>& x,
                  std::vector<std::vector<float>>& y,
                  float** dev_x,
                  float** dev_y, 
                  float** dev_z)
{
    size_t x_size = x.size();
    size_t x0_size = x[0].size();
    size_t xflat_size = x_size * x0_size;
    float* x_arr = (float*) malloc(xflat_size * sizeof(float));
    for (size_t i = 0; i < x_size; ++i)
        for (size_t j = 0; j < x0_size; ++j)
            x_arr[i*x0_size + j] = x[i][j];

    size_t y_size = y.size();
    size_t y0_size = y[0].size();
    size_t yflat_size = y_size * y0_size;
    float* y_arr = (float*) malloc(yflat_size * sizeof(float));
    for (size_t i = 0; i < y_size; ++i)
        for (size_t j = 0; j < y0_size; ++j)
            y_arr[i*y0_size + j] = y[i][j];
    CopyData(x_arr, xflat_size, sizeof(float), dev_x);
    CopyData(y_arr, yflat_size, sizeof(float), dev_y);
    size_t z_size = x_size * y0_size;
    float* z = (float*)malloc(z_size * sizeof(float));
    memset(z, 0, sizeof(float) * z_size);
    CopyData(z, z_size, sizeof(float), dev_z);
}

void cudaToMatrix(float** dev_z,
                  std::vector<std::vector<float>>& z)
{
    //TODO
}

void cudaMatrixMultiply(std::vector<std::vector<float>> x,
                        std::vector<std::vector<float>> y,
                        std::vector<std::vector<float>>& z)
{
    float* dev_x;
    float* dev_y;
    float* dev_z;
    matrixToCuda(x, y, &dev_x, &dev_y, &dev_z);
    //cuda_matrix_mult<<<x[0].size(), y.size()>>>(x[0].size(), y.size(), dev_x, dev_y, dev_z);
    size_t threads=64;
    dim3 dimGrid(x[0].size(), x.size(), 1);
    dim3 dimBlock(threads, threads, 1);
    size_t shared = threads*sizeof(float);
    cuda_matrix_mult<<<dimGrid, dimBlock, shared>>>(x[0].size(), y.size(), dev_x, dev_y, dev_z);
    //TODO
    cudaToMatrix(&dev_z, z);
    printf("We made it out\n");
}


/*
void modelToGPU(Sequential& model)
{
    //std::vector<LinearLayer> layers= model.getLayers();
}
*/

// We want to keep everything in the cuda memory so redo a lot of stuff
/*
template <class T>
__global__ void
cuda_forward(T* inputs, T* outputs)
{
}
*/

