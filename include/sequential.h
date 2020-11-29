#include <iostream>
#include "nn.h"
#include "types.h"
#include "math_funcs.h"
#include <vector>

class Sequential
{

	private:
		std::vector<LinearLayer> layers;
                size_t batch_size;
		math_funcs math;

	public:
		Sequential();
		std::vector<std::vector<float>> forward(std::vector<std::vector<float>>);
		void backward(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
		inline void add(LinearLayer l){layers.emplace_back(l);};

		void trainIteration(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
		std::vector<std::vector<float>> lossFunctionDerivative(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
                inline void setBatchSize(size_t bs){this->batch_size = bs;};
                // Helpers
		void printModel();
		inline std::vector<LinearLayer> getLayers(){return this->layers;};
                inline size_t getBatchSize(){return this->batch_size;};
};
