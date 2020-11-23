#include <iostream>
#include "nn.h"
#include "types.h"
#include "math_funcs.h"
#include "sequential.h"
#include <vector>
#include "utils.h"

Sequential::Sequential(){
}

std::vector<std::vector<float>> Sequential::forward(std::vector<std::vector<float>> inputs)
{
	//std::cout << "\n Forward pass \n\n";
	std::vector<std::vector<float>> current_input(inputs.begin(), inputs.end());

        // DO NOT PARALLELIZE
	for (size_t i = 0; i < layers.size(); ++i)
	{
		layers[i].forward(current_input);
		current_input = layers[i].getActivations();
	}
	return current_input;
}

void Sequential::backward(std::vector<std::vector<float>> error, std::vector<std::vector<float>> inputs)
{
	//output layer
	std::vector<std::vector<float>> current_weights;
        for (size_t i = error[0].size(); i != 0; --i) 
	{
		std::vector<float> w{1.0};
		current_weights.emplace_back(w);
	}
	//computing deltas
	std::vector<std::vector<float>> current_error(error.begin(), error.end());
        // Need signed integer value here!
        // DO NOT PARALLELIZE
	for(int i = layers.size()-1; i >= 0; i--)
	{
		layers[i].computeDeltas(current_error, current_weights);
		current_error = layers[i].getDeltas();
		current_weights = layers[i].getWeights();
	}

	//updating the weights for all but the first (input) layer
	
        // We can parallelize this one because we already have the weights
        #pragma omp parallel for schedule(static)
        for (size_t i = 1; i < layers.size(); ++i)
		layers[i].updateWeightsLegacy(layers[i-1].getActivations());
	//updating the first layer
	layers[0].updateWeightsLegacy(inputs);
	
}

void Sequential::trainIteration(std::vector<std::vector<float>> training_inputs, std::vector<std::vector<float>> labels)
{
	//std::cout << "\nTraining Inputs: \n";
	std::vector<std::vector<float>> preds = this->forward(training_inputs);

	//std::cout << "\nPredictions: \n";
	this->backward(lossFunctionDerivative(preds, labels), training_inputs);
}

std::vector<std::vector<float>> Sequential::lossFunctionDerivative(std::vector<std::vector<float>> predictions, std::vector<std::vector<float>> labels)
{

	std::vector<std::vector<float>> errors(predictions.size());
        //math.matrix_sub(labels, predictions, errors);
        #pragma omp parallel for schedule(static)
	for (size_t i = 0; i < predictions.size(); i++)
	{
		std::vector<float> error(predictions[i].size());
		for (size_t j = 0; j < predictions[i].size(); j++)
                        error[j] = labels[i][j] - predictions[i][j];
                errors[i] = error;
	}
	return errors;
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

