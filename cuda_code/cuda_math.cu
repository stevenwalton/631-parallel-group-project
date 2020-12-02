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
                            std::vector<int>& labels,
                            size_t batch_size,
                            float** dev_features,
                            int** dev_labels,
                            float** dev_predictions)
{
    // Flatten vector of vectors and push to arrays
    matrix2Cuda(features, dev_features);
    vector2Cuda(labels, dev_labels);

    // TODO
    // Check that preds is the correct shape and size
    float* preds = (float*)malloc(batch_size * sizeof(float));
    memset(preds, 0, batch_size * sizeof(float));
    CopyData(preds, batch_size, sizeof(float), dev_predictions);
}

template <class T>
void matrix2Cuda(std::vector<std::vector<T>>& m, T** dev_m)
{
	size_t m_rows = m.size();
	size_t m_cols = m[0].size();
	size_t m_total = m_rows * m_cols;
	T* arr = (T*)malloc(m_total * sizeof(T));
	for (size_t i = 0; i < m_rows; ++i)
        	for (size_t j = 0; j < m_cols; ++j)
            		arr[i*m_cols + j] = m[i][j];
    	assert(arr);
	CopyData(arr, m_total, sizeof(T), dev_m);
}

template <class T>
void vector2Cuda(std::vector<T>& v, T** dev_v)
{
        size_t v_total = v.size();
        T* arr = (T*)malloc(v_total * sizeof(T));
        for (size_t i = 0; i < v_total; ++i)
                        arr[i] = v[i];
        assert(arr);
        CopyData(arr, v_total, sizeof(T), dev_v);
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
    matrix2Cuda(x, &dev_x);
    matrix2Cuda(y, &dev_y);
    matrix2Cuda(z, &dev_z);
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

