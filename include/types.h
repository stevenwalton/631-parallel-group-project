#ifndef TYPES_H
#define TYPES_H

#include <vector>

struct node
{
   	float bias;
	std::vector<float> weight;
	//std::vector<float> weight_grad;
	//All these vectors have batch_size length
	std::vector<float> activation;
	std::vector<float> delta; //gradient
	std::vector<float> error;
};

#endif
