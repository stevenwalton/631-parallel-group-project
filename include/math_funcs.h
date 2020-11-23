#ifndef MATH_FUNCS
#define MATH_FUNCS 1
/*
 * Macs are super dumb and we're required to have this in a class or you get a
 * duplicate symbol error.
 *
 * If you want the function to not be within the object declaration it must go
 * into a .cpp fine (I hate Macs...)
 */

#include <vector>
#include <cmath>
#include "types.h"
#include <cassert>
class math_funcs
{
    public:
        float sigmoid(float x){return std::exp(x)/ (std::exp(x) + 1.);}
        float derivative_sigmoid(float x){return x * (1. - x);}
        float dot_product(std::vector<float>, std::vector<float>);
        float dot_product(float, std::vector<float>, float, std::vector<float>);
        float vector_sum(std::vector<float>);
        void scale_vector(float, std::vector<float>&);

	inline float unit_random(){ return ((float)rand())/((float)RAND_MAX); };
};


#endif
