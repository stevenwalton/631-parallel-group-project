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

class math_funcs
{
    public:
        float sigmoid(float x){return std::exp(x)/ (std::exp(x) + 1.);}
        float derivative_sigmoid(float x){return x * (1. - x);}
        void MSE(std::vector<struct node>& x, std::vector<struct node> y)
            {for (node i : x) {
                    for (node j : y) {
                        i.error += pow((j.activation - j.activation),2.);
                    } } };
};


#endif
