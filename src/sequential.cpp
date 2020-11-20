#include <iostream>
#include "nn.h"
#include "types.h"
#include "math_funcs.h"
#include "sequential.h"
#include <vector>
#include "utils.h"

Sequential::Sequential(){
}

std::vector<float> Sequential::forward(std::vector<float> inputs)
{
	std::vector<float> current_input(inputs.begin(), inputs.end());
	for (size_t i = 0; i < layers.size(); ++i)
	{
		std::cout << "\nLayer " << i << " weights: \n";
		layers[i].printWeights();
		layers[i].forward(current_input);
		current_input = layers[i].getActivations();
	}
	//returning the last layer's activations
	return current_input;
}

void Sequential::backward(std::vector<float> error, std::vector<float> inputs)
{
	//Creating the fake 'weights' vector of vectors for the 
	//output layer
	std::vector<std::vector<float>> current_weights;
	//for(float f : error)
        for (size_t i = error.size(); i != 0; --i) 
	{
		std::vector<float> w{1.0};
		current_weights.emplace_back(w);
	}
	//computing deltas
	std::vector<float> current_error(error.begin(), error.end());
	//careful with size_t as int leads to unexpected behaviour
	//apparently can't go negative
	for(int i = layers.size()-1; i >= 0; i--)
	{
		layers[i].computeDeltas(current_error, current_weights);
		current_error = layers[i].getDeltas();
		current_weights = layers[i].getWeights();
	}
	//updating the weights for all but the first (input) layer
	for(size_t i = layers.size()-1; i > 0; i--)
	{
		layers[i].updateWeights(layers[i-1].getActivations());
	}
	//updating the first layer
	layers[0].updateWeights(inputs);
}

void Sequential::trainIteration(std::vector<float> training_inputs, std::vector<float> labels)
{
	std::cout << "Training inputs: \n";
	printFloatVector(training_inputs);
	std::vector<float> preds = this->forward(training_inputs);
	std::cout << "Predictions inputs: \n";
	printFloatVector(preds);
	this->backward(lossFunctionDerivative(preds, labels), training_inputs);
}

std::vector<float> Sequential::lossFunctionDerivative(std::vector<float> predictions, std::vector<float> labels)
{
	std::vector<float> error;
	for (size_t i = 0; i < labels.size(); ++i)
		error.emplace_back(labels[i] - predictions[i]);
	return error;
}

/***************** Helper functions **********************************/
void Sequential::printModel(){
	std::cout << "Model has " << this->layers.size() << " layers: \n";
	for (size_t i = 0; i < layers.size(); ++i)
	{
	std::cout << "Layer "<< i << ": (" << layers[i].getNumInputs() << ", " << layers[i].getNumNeurons() << ")\n";
	}
	std::cout << "\n";

}

