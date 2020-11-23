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
