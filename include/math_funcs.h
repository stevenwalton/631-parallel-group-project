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
        static float sigmoid(float x){return std::exp(x)/ (std::exp(x) + 1.);}
        static float derivative_sigmoid(float x){return x * (1. - x);}
        float dot_product(std::vector<float>, std::vector<float>);
        float dot_product(float, std::vector<float>, float, std::vector<float>);
        float vector_sum(std::vector<float>);
        void scale_vector(float, std::vector<float>&);
        void vector_add(std::vector<float>,
                        std::vector<float>,
                        std::vector<float>&);
        void vector_sub(std::vector<float>,
                        std::vector<float>,
                        std::vector<float>&);

        void matrix_add(std::vector<std::vector<float>>,
                        std::vector<std::vector<float>>,
                        std::vector<std::vector<float>>&);
        void matrix_add(float,
                        std::vector<std::vector<float>>,
                        float,
                        std::vector<std::vector<float>>,
                        std::vector<std::vector<float>>&);
        void matrix_sub(std::vector<std::vector<float>>,
                        std::vector<std::vector<float>>,
                        std::vector<std::vector<float>>&);
	void matrix_mult(std::vector<std::vector<float>>,
                         std::vector<std::vector<float>>,
                         std::vector<std::vector<float>>&);
	void matrix_plus_vec(std::vector<std::vector<float>>&, std::vector<float>);
	void map_function(std::vector<std::vector<float>> &, 
			  float func (float));
	std::vector<std::vector<float>> matrix_transpose(std::vector<std::vector<float>>);
	void elem_matrix_mult(std::vector<std::vector<float>>,
                            std::vector<std::vector<float>>,
                            std::vector<std::vector<float>>&);
	void transposed_element_matrix_mult(std::vector<std::vector<float>>,
                            std::vector<std::vector<float>>,
                            std::vector<std::vector<float>>&,
                            float func(float));
	void transposed_element_matrix_mult(std::vector<std::vector<float>>,
                            std::vector<std::vector<float>>,
                            std::vector<std::vector<float>>&);
	inline float unit_random(){ return ((float)rand())/((float)RAND_MAX); };
};


#endif

