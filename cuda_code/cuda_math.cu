#include <iostream>
#include <cstdio>
#include <vector>
#include <cassert>
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "helper_cuda.h"
#include "cooperative_groups.h"

#include "cuda_math.h"

// Kernels go up top
//Luis version
//Basically launching one block for each element in the 
//result matrix
template <class T>
__global__ void
cuda_matrix_mult(size_t inner_size, size_t y_cols, T* x, T* y, T* z)
{
    	extern __shared__ double shared_mem[];
    	int row = blockIdx.x;
    	int col = blockIdx.y;
    	int n_threads = blockDim.x;
    	int tid = threadIdx.x;

	size_t i;
    	//clear the shared memory
    	shared_mem[tid] = 0.0;
    	__syncthreads();

	//printf("\nrow = %d col = %d\n", row, col);

	for (i = 0 + tid; i < inner_size; i+=n_threads)
	{
		shared_mem[tid] += x[row * inner_size + i] * y[i * y_cols + col];
	}
	__syncthreads();

	if(tid == 0){
		z[row*y_cols + col] = 0.0;
		for (i = 0; i < n_threads; i++)
			z[row*y_cols + col] += shared_mem[i];
	}
}

//one block per row in the resulting matrix
//each thread will take care of one of the columns
//thus, we need to have as many threads as columns
//Luis' important note: 
//of course this limits the size of the matrix that 
//can actually be obtained to N X 1024 which should
//be enough for our purposes
template <class T>
__global__ void
cuda_matrix_mult_v2(size_t inner_size, T* x, T* y, T* z)
{
        extern __shared__ double shared_mem[];
        int row = blockIdx.x;
        int tid = threadIdx.x;  //basically the col
        int n_cols = blockDim.x;

        size_t i;
        //clear the shared memory
        shared_mem[tid] = 0.0;
        __syncthreads();

        //printf("\nrow = %d col = %d\n", row, col);

        for (i = 0; i < inner_size; ++i)
        {
                shared_mem[tid] += x[row * inner_size + i] * y[i * n_cols + tid];
        }
        __syncthreads();

	//no reduction or further synchronization needed, 
	//simply write to output array
	z[row * n_cols + tid] = shared_mem[tid];
}

/*Steven's version
   template <class T>
__global__ void
cuda_matrix_mult(size_t x_size, size_t y_size,
                 T* x, T* y, T* z)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;
    if ( row < x_size && col < y_size)
    {
        printf("\nrow = %d col = %d\n", row, col);
        T tmp = 0;
        for (size_t k = 0; k < y_size; ++k)
        {
            tmp += x[row*y_size+ k] * y[k*y_size + col];
            printf("%f * %f\n", x[row*y_size+ k], y[k*y_size + col]);
        }
        z[row*y_size+ col] = tmp;
    }
}
*/
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

  // Allocate pinned memory on host (for faster HtoD copy)
  T* h_in_pinned = NULL;
  checkCudaErrors(cudaMallocHost((void**) &h_in_pinned, N * dsize));
  assert(h_in_pinned);
  memcpy(h_in_pinned, input, N * dsize);

  // copy data
  checkCudaErrors(cudaMalloc((void**) d_in, N * dsize));
  //checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpy(*d_in, h_in_pinned,
                             N * dsize, cudaMemcpyHostToDevice));
  //checkCudaErrors(cudaEventRecord(stop, 0));
  //checkCudaErrors(cudaEventSynchronize(stop));
  //checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  //printf("  Pinned Device to Host bandwidth (GB/s): %f\n",(N * dsize) * 1e-6 / elapsedTime);

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

template <class T>
void cuda2Matrix(T* dev_z, std::vector<std::vector<T>>& z)
{
    	size_t z_rows = z.size();
	size_t z_cols = z[0].size();
	size_t z_total = z_rows * z_cols;
	//creating an array to hold the data 
	T* arr = (T*)malloc(z_total * sizeof(T));

	//copying from GPU to CPU
    	checkCudaErrors(cudaMemcpy(arr, dev_z, sizeof(T) * z_total, cudaMemcpyDeviceToHost));
	
	//copying data into vector
	for (size_t i = 0; i < z_rows; ++i)
	{
                for (size_t j = 0; j < z_cols; ++j)
		{
                        z[i][j] = arr[i*z_cols + j];
			//std::cout << z[i][j] << " ";
		}
		//std::cout << std::endl;
	}
	//freeing memory
	free(arr);
}

template <class T>
void cuda2Vector(T** dev_z, std::vector<T>& z)
{
        size_t z_total = z.size();
        //creating an array to hold the data
        T* arr = (T*)malloc(z_total * sizeof(T));

        //copying from GPU to CPU
        checkCudaErrors(cudaMemcpy(arr, dev_z, sizeof(T) * z_total, cudaMemcpyDeviceToHost));

        //copying data into vector
        for (size_t i = 0; i < z_total; ++i)
	{
                        z[i] = arr[i];
			std::cout << arr[i];
	}
	std::cout << "\n";
        //freeing memory
        free(arr);
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
    size_t threads=32;
    
    //One block for each element in the result matrix
    dim3 dimGrid(x.size(), y[0].size(), 1);
    //Each block has 32 x 32 threads = 1024
    dim3 dimBlock(threads, 1, 1);
    //Using some shared memory, one float for each thread.
    unsigned int shared = threads * sizeof(float);
    cuda_matrix_mult<<<dimGrid, dimBlock, shared>>>(x[0].size(), y[0].size(), dev_x, dev_y, dev_z);
    //TODO
    cuda2Matrix(dev_z, z);
    //printf("We made it out\n");
}

void cudaMatrixMultiplyv2(std::vector<std::vector<float>> x,
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
    size_t threads=y[0].size();
    //1024 is the max number of threads per block
    assert(threads <= 1024);

    //One block for each row in the resulting matrix
    dim3 dimGrid(x.size(), 1, 1);
    //Each block has as many threads as columns in the resulting matrix
    dim3 dimBlock(threads, 1, 1);
    //Using some shared memory, one float for each thread.
    unsigned int shared = threads * sizeof(float);
    cuda_matrix_mult_v2<<<dimGrid, dimBlock, shared>>>(x[0].size(), dev_x, dev_y, dev_z);
    //TODO
    cuda2Matrix(dev_z, z);
    //printf("We made it out\n");
}
