#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <omp.h>
#include "math_funcs.h"
#include<algorithm>

float math_funcs::dot_product(std::vector<float> x, std::vector<float> y)
{
    //assert(x.size() == y.size() || !(fprintf(stderr, "Dot products are not the same size: %lu vs %lu\n", x.size(), y.size())));
    assert(x.size() == y.size());
    float product = 0;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        product += x[i] * y[i];
    return product;

}

float math_funcs::dot_product(float a, std::vector<float> x, float b, std::vector<float> y)
{
    //assert(x.size() == y.size() || !(fprintf(stderr, "Dot products are not the same size: %lu vs %lu\n", x.size(), y.size())));
    assert(x.size() == y.size());
    float product = 0;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        product += (a * x[i]) * (b * y[i]);
    return product;

}

float math_funcs::vector_sum(std::vector<float> v)
{
    float sum = 0;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (size_t i = 0; i < v.size(); ++i)
        sum += v[i];
    return sum;
}

float math_funcs::vector_mean(std::vector<float> v)
{
	return vector_sum(v)/v.size();
}

void math_funcs::scale_vector(float a, std::vector<float> &v)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < v.size(); ++i)
        v[i] *= a;
}

void math_funcs::vector_add(std::vector<float> x, 
                            std::vector<float> y,
                            std::vector<float>& z)
{
    //assert(x.size() == y.size() || !(fprintf(stderr, "Cannot add vectors because they are not the same size: %lu, %lu\n", x.size(), y.size())));
    assert(x.size() == y.size());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        z[i] = x[i] + y[i];
}

void math_funcs::vector_sub(std::vector<float> x, 
                            std::vector<float> y,
                            std::vector<float>& z)
{
    //assert(x.size() == y.size() || !(fprintf(stderr, "Cannot add vectors because they are not the same size: %lu, %lu\n", x.size(), y.size())));
    assert(x.size() == y.size());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        z[i] = x[i] - y[i];
}

void math_funcs::matrix_add(std::vector<std::vector<float>> x,
                            std::vector<std::vector<float>> y,
                            std::vector<std::vector<float>>& z)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        for (size_t j = 0; j < x[0].size(); ++j)
            z[i][j] = x[i][j] + y[i][j];
}

void math_funcs::matrix_add(float a,
                            std::vector<std::vector<float>> x,
                            float b,
                            std::vector<std::vector<float>> y,
                            std::vector<std::vector<float>>& z)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        for (size_t j = 0; j < x[0].size(); ++j)
            z[i][j] = (a * x[i][j]) + (b * y[i][j]);
}

void math_funcs::matrix_sub(std::vector<std::vector<float>> x,
                            std::vector<std::vector<float>> y,
                            std::vector<std::vector<float>>& z)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
    {
        z[i].resize(x[i].size());
        for (size_t j = 0; j < x[0].size(); ++j)
            z[i][j] = x[i][j] - y[i][j];
    }
}

void math_funcs::matrix_mult(std::vector<std::vector<float>> x,
                            std::vector<std::vector<float>> y,
                            std::vector<std::vector<float>>& z)
{
    //assert(x[0].size() == y.size() || !(fprintf(stderr, "Incorrect matrix dimensions: %lu x %lu, %lu x %lu\n", x.size(), x[0].size(), y.size(), y[0].size)));
    assert(x[0].size() == y.size());
    //std::cout << x[0].size() << " " << y.size() << std::endl;
    #pragma omp parallel for schedule(static)
    for (size_t i =0; i < x.size(); ++i) //each row in x
    {
    	for (size_t j = 0; j < y[0].size(); ++j) //each column in y
    	{
        	for (size_t k = 0; k < x[0].size(); ++k) //all columns in x
        	{
			//std::cout << x[i][k] << " * " << y[k][j] << std::endl;
            		z[i][j] += x[i][k] * y[k][j];
        	}
    	}
    }
}

void math_funcs::matrix_plus_vec(std::vector<std::vector<float>> &x, std::vector<float> b)
{
	//assert(x[0].size() == b.size() || !(fprintf(stderr, "Cannot add vector to matrix due to incorrect dimensions: %lu %lu\n", x[0].size(), b.size())));
	assert(x[0].size() == b.size());
	for (size_t i = 0; i < x.size(); ++i)
	{
		for (size_t j = 0; j < x[i].size(); ++j)
			x[i][j] += b[j];
	}
}

void math_funcs::map_function(std::vector<std::vector<float>> &x, float func (float))
{
	for (size_t i = 0; i < x.size(); ++i)
		for (size_t j = 0; j < x[0].size(); ++j)
			x[i][j] = func(x[i][j]);
}

std::vector<std::vector<float>> math_funcs::matrix_transpose(std::vector<std::vector<float>> x)
{
	std::vector<std::vector<float>> result;

	for (size_t j = 0; j < x[0].size(); ++j)
	{	
		std::vector<float> v;
		for (size_t i = 0; i < x.size(); ++i)
			v.emplace_back(x[i][j]);
		result.emplace_back(v);
	}
	return result;
}

void math_funcs::elem_matrix_mult(std::vector<std::vector<float>> x,
                            std::vector<std::vector<float>> y,
                            std::vector<std::vector<float>>& z)
{
        //assert(x.size() == y.size() && x[0].size() == y[0].size() || !(fprintf(stderr, "Incorrect matrix dimensions: %lu x %lu, %lu x %lu\n", x.size(), x[0].size(), y.size(), y[0].size)));
	assert(x.size() == y.size() && x[0].size() == y[0].size());
       	std::cout << x.size() << " " << y.size() << std::endl;
	for (size_t i = 0; i < x.size(); ++i)
        {
                for (size_t j = 0; j < x[0].size(); ++j)
                {
                        z[i][j] = x[i][j] * y[i][j];
                }
        }
}


//this function multiplies the elements of a matrix by its corresponding element in a transposed matrix
//i.e a 32 x 5 matrix is multiplied per element by a 5 x 32 matrix and the resulting matrix will be 5 x 32
//I use this when multiplying the errors by the derivative of the activations to obtain the deltas.
//z will have the same dimensions as y
void math_funcs::transposed_element_matrix_mult(std::vector<std::vector<float>> x,
                            std::vector<std::vector<float>> y,
                            std::vector<std::vector<float>>& z)
{
	//assert(x[0].size() == y.size() && x.size() == y[0].size() || !(fprintf(stderr, "Incorrect matrix dimensions: %lu x %lu, %lu x %lu\n", x.size(), x[0].size(), y.size(), y[0].size)));
	assert(x[0].size() == y.size() && x.size() == y[0].size());
	for (size_t i = 0; i < x.size(); ++i)
	{
		for (size_t j = 0; j < x[0].size(); ++j)
		{
			z[j][i] = x[i][j] * y[j][i];
		}
	}
}

void math_funcs::transposed_element_matrix_mult(std::vector<std::vector<float>> x,
                            std::vector<std::vector<float>> y,
                            std::vector<std::vector<float>>& z,
			    float func(float))
{
        //assert(x[0].size() == y.size() && x.size() == y[0].size() || !(fprintf(stderr, "Incorrect matrix dimensions: %lu x %lu, %lu x %lu\n", x.size(), x[0].size(), y.size(), y[0].size)));
	assert(x[0].size() == y.size() && x.size() == y[0].size());
        for (size_t i = 0; i < x.size(); ++i)
        {
                for (size_t j = 0; j < x[0].size(); ++j)
                {
                        z[j][i] = func(x[i][j]) * y[j][i];
                }
        }
}
