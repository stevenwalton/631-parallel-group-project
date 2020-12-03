#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <omp.h>
#include "math_funcs.h"
#include<algorithm>

/*
float math_funcs::dot_product(std::vector<float> x, std::vector<float> y)
{
    //assert(x.size() == y.size() || !(fprintf(stderr, "Dot products are not the same size: %lu vs %lu\n", x.size(), y.size())));
    assert(x.size() == y.size());
    float product = 0;
    //#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        product += x[i] * y[i];
    return product;

}

float math_funcs::dot_product(float a, std::vector<float> x, float b, std::vector<float> y)
{
    //assert(x.size() == y.size() || !(fprintf(stderr, "Dot products are not the same size: %lu vs %lu\n", x.size(), y.size())));
    assert(x.size() == y.size());
    float product = 0;
    //#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        product += (a * x[i]) * (b * y[i]);
    return product;

}
*/

float math_funcs::vector_sum(std::vector<float> v)
{
    float sum = 0;
    for (size_t i = 0; i < v.size(); ++i)
        sum += v[i];
    return sum;
}

void math_funcs::scale_vector(float a, std::vector<float> &v)
{
    //#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < v.size(); ++i)
        v[i] *= a;
}

void math_funcs::scale_matrix(std::vector<float> a, std::vector<std::vector<float>> &m)
{
    /*
     * Not used
     */
    assert(a.size() == m.size());
    for(size_t i = 0; i < m.size(); ++i)
        scale_vector(a[i], m[i]);
}

void math_funcs::scale_matrix(float a, std::vector<std::vector<float>> &m)
{
    /*
     * OMP: No change, little work
     */
    for(size_t i = 0; i < m.size(); ++i)
        scale_vector(a, m[i]);
}

//this method is similar to the last one but scales by 1/a 
void math_funcs::inverse_scale_matrix(std::vector<float> a, std::vector<std::vector<float>> &m)
{
        assert(a.size() == m.size());
        for(size_t i = 0; i < m.size(); ++i)
            scale_vector(1.0/a[i], m[i]);
}

void math_funcs::inverse_scale_matrix(float a, std::vector<std::vector<float>> &m)
{
        for(size_t i = 0; i < m.size(); ++i)
            scale_vector(1.0/a, m[i]);
}

void math_funcs::vector_add(std::vector<float> x, 
                            std::vector<float> y,
                            std::vector<float>& z)
{
    assert(x.size() == y.size());
    /*
     * OMP Slight improvement
     * static is worse, keep dynamic
     */
    #pragma omp parallel for 
    for (size_t i = 0; i < x.size(); ++i)
        z[i] = x[i] + y[i];
}

void math_funcs::vector_sub(std::vector<float> x, 
                            std::vector<float> y,
                            std::vector<float>& z)
{
    assert(x.size() == y.size());
    /*
     * OMP Slight improvement
     * static is worse, keep dynamic
     */
    #pragma omp parallel for 
    for (size_t i = 0; i < x.size(); ++i)
        z[i] = x[i] - y[i];
}

void math_funcs::matrix_add(std::vector<std::vector<float>> x,
                            std::vector<std::vector<float>> y,
                            std::vector<std::vector<float>>& z)
{
    /*
     * OMP: Good improvement
     * keep static
     */
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
    /*
     * OMP: Good improvement
     * keep static
     */
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        for (size_t j = 0; j < x[0].size(); ++j)
            z[i][j] = (a * x[i][j]) + (b * y[i][j]);
}

void math_funcs::matrix_sub(std::vector<std::vector<float>> x,
                            std::vector<std::vector<float>> y,
                            std::vector<std::vector<float>>& z)
{
    /*
     * OMP: Not enough work
     * static better than dynamic but none better than omp
     */
    //#pragma omp parallel for 
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
    size_t x_size = x.size();
    size_t x0_size = x[0].size();
    size_t y0_size = y[0].size();
    /*
     * blocking makes no difference because matrix sizes/order
     * Div by 2 block size is faster than div by 4 and div by 8
     */
    /*
    size_t x_block = 1;
    size_t x0_block = 1;
    size_t y0_block = 1;
    if (x_size % 2 == 0) x_block = x_size/2;
    if (x0_size% 2 == 0) x0_block = x0_size / 2;
    if (y0_size % 2 == 0) y0_block = y0_size / 2;
    //printf("x %d, x0 %d, xb: %d, y %d, y0 %d\n", x_size, x0_size, x_block, y.size(), y0_size);
    assert(x[0].size() == y.size());
    for (size_t xb = 0; xb < x_size; xb += x_block)
        for (size_t x0b = 0; x0b < x0_size; x0b += x0_block)
            for (size_t y0b = 0; y0b < y0_size; y0b += y0_block)
                #pragma omp parallel for schedule(static)
                for (size_t i = xb; i < xb + x_block; ++i)
                    for (size_t k = x0b; k < x0b + x0_block; ++k)
                        for (size_t j = y0b; j < y0b + y0_block; ++j)
                            z[i][j] += x[i][k] * y[k][j];
    */
    /*
     *  (Naive)
     *  Actually slower. 6s w/ 32 cores vs 2.1s with naive
     */
    /*
    #pragma omp parallel for schedule(static)
    for (size_t i =0; i < x_size; ++i) //each row in x
    	for (size_t k = 0; k < x0_size; ++k) //each column in y
            for (size_t j = 0; j < y0_size; ++j) //all columns in x
	    {
                    z[i][j] += x[i][k] * y[k][j];
	    }
    */
    /*
     * OMP: Good improvement
     * slight increase when static
     */
    /*
     * Smart ordering
     */
    
    #pragma omp parallel for schedule(static)
    for (size_t i =0; i < x_size; ++i) //each row in x
    	for (size_t j = 0; j < y0_size; ++j) //each column in y
	{
		z[i][j] = 0.0;
		for (size_t k = 0; k < x0_size; ++k) //all columns in x
                	z[i][j] += x[i][k] * y[k][j];
	}
		
}

void math_funcs::matrix_plus_vec(std::vector<std::vector<float>> &x, std::vector<float> b)
{
    assert(x[0].size() == b.size());
    /*
     * OMP: Slight improvement
     */
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        for (size_t j = 0; j < x[i].size(); ++j)
            x[i][j] += b[j];
}

void math_funcs::map_function(std::vector<std::vector<float>> &x, float func (float))
{
    /*
     * OMP: Good improvement!
     */
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        for (size_t j = 0; j < x[0].size(); ++j)
            x[i][j] = func(x[i][j]);
}

std::vector<std::vector<float>> math_funcs::matrix_transpose(std::vector<std::vector<float>> x)
{
    /*
     * OMP slows this down
     */

    size_t x_size = x.size();
    size_t x0_size = x[0].size();
    std::vector<std::vector<float>> result(x0_size);
    std::vector<float> v(x_size);
    //#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < x0_size; ++j)
    {	
        for (size_t i = 0; i < x_size; ++i)
            v[i] = x[i][j];
        result[j] = v;
    }
    return result;
}

void math_funcs::elem_matrix_mult(std::vector<std::vector<float>> x,
                            std::vector<std::vector<float>> y,
                            std::vector<std::vector<float>>& z)
{
    assert(x.size() == y.size() && x[0].size() == y[0].size());
    std::cout << x.size() << " " << y.size() << std::endl;
    /*
     * OMP: Good improvement 
     */
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        for (size_t j = 0; j < x[0].size(); ++j)
            z[i][j] = x[i][j] * y[i][j];
}


//this function multiplies the elements of a matrix by its corresponding element in a transposed matrix
//i.e a 32 x 5 matrix is multiplied per element by a 5 x 32 matrix and the resulting matrix will be 5 x 32
//I use this when multiplying the errors by the derivative of the activations to obtain the deltas.
//z will have the same dimensions as y
void math_funcs::transposed_element_matrix_mult(std::vector<std::vector<float>> x,
                            std::vector<std::vector<float>> y,
                            std::vector<std::vector<float>>& z)
{
    /*
     * OMP Slows down: not enough work
     */
    assert(x[0].size() == y.size() && x.size() == y[0].size());
    // So little work this optimization does almost nothing
    size_t x_size = x.size();
    size_t x0_size = x[0].size();
    /*
     * OMP: slight improvement
     */
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x_size; ++i)
        for (size_t j = 0; j < x0_size; ++j)
            z[j][i] = x[i][j] * y[j][i];
}

void math_funcs::transposed_element_matrix_mult(std::vector<std::vector<float>> x,
                            std::vector<std::vector<float>> y,
                            std::vector<std::vector<float>>& z,
			    float func(float))
{
    assert(x[0].size() == y.size() && x.size() == y[0].size());
    /*
     * OMP: Very slight improvement
     * keep static
     */
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        for (size_t j = 0; j < x[0].size(); ++j)
            z[j][i] = func(x[i][j]) * y[j][i];
}
