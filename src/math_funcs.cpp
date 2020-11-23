#include <stdio.h>
#include <assert.h>
#include <omp.h>
#include "math_funcs.h"

float math_funcs::dot_product(std::vector<float> x, std::vector<float> y)
{
    assert(x.size() == y.size() || !(fprintf(stderr, "Dot products are not the same size: %lu vs %lu\n", x.size(), y.size())));
    float product = 0;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        product += x[i] * y[i];
    return product;

}

float math_funcs::dot_product(float a, std::vector<float> x, float b, std::vector<float> y)
{
    assert(x.size() == y.size() || !(fprintf(stderr, "Dot products are not the same size: %lu vs %lu\n", x.size(), y.size())));
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
    assert(x.size() == y.size() || !(fprintf(stderr, "Cannot add vectors because they are not the same size: %lu, %lu\n", x.size(), y.size())));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < x.size(); ++i)
        z[i] = x[i] + y[i];
}

void math_funcs::vector_sub(std::vector<float> x, 
                            std::vector<float> y,
                            std::vector<float>& z)
{
    assert(x.size() == y.size() || !(fprintf(stderr, "Cannot add vectors because they are not the same size: %lu, %lu\n", x.size(), y.size())));
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
