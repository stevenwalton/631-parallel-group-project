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

		void add(LinearLayer);

		std::vector<float> forward(std::vector<float>);

		void backward(std::vector<float>, std::vector<float>);

		void trainIteration(std::vector<float>, std::vector<float>);

		std::vector<float> lossFunctionDerivative(std::vector<float>, std::vector<float>);

		std::vector<LinearLayer> getLayers();

		void printModel();
};
