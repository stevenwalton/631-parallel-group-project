#include <iostream>
#include "nn.h"
#include "types.h"
#include "math_funcs.h"
#include <vector>

class Sequential
{

	private:
		std::vector<LinearLayer> layers;

	public:
		Sequential();
		std::vector<float> forward(std::vector<float>);
		void backward(std::vector<float>, std::vector<float>);
		inline void add(LinearLayer l){layers.emplace_back(l);};

		void trainIteration(std::vector<float>, std::vector<float>);
		std::vector<float> lossFunctionDerivative(std::vector<float>, std::vector<float>);
                // Helpers
		void printModel();
		inline std::vector<LinearLayer> getLayers(){return this->layers;};
};
