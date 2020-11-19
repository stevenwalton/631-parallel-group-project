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
        /*void MSE(std::vector<struct node>& x, std::vector<struct node> y)
        {
		    for (node i : x) {
                    	i.error = 0;
                    	for (node j : y) {
                        	i.error += pow(i.activation,2) - pow(j.activation,2); 
                    	}
                    	sqrt(i.error);
            	} 
	};*/
	float unit_random(){
    		return ((float)rand())/((float)RAND_MAX);
	};
	float dot_product(std::vector<float> a, std::vector<float> b)
	{
		assert(a.size() == b.size());
		float product = 0.0;
		for (size_t i = 0; i < a.size(); i++)
			product += a[i] * b[i];
		return product;
	};
	float vector_sum(std::vector<float> v)
	{
		float sum = 0.0;
		for (float f : v)
			sum += f;
		return sum;
	}
};


#endif
