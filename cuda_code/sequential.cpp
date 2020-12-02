#include <iostream>
#include "nn.h"
#include "types.h"
#include "math_funcs.h"
#include "sequential.h"
#include "utils.h"
#include <vector>
#include "utils.h"
#include "cuda_math.h"

Sequential::Sequential(){
}

std::vector<std::vector<float>> Sequential::forward(std::vector<std::vector<float>> inputs)
{
	//std::cout << "\n Forward pass \n\n";
	std::vector<std::vector<float>> current_input(inputs.begin(), inputs.end());

	for (size_t i = 0; i < layers.size(); ++i)
	{
		layers[i].forward(current_input);
		current_input = layers[i].getActivations();
	}
	//returning the last layer's activations
	//for(std::vector<float> in : current_input)
        //               printFloatVector(in);
	return current_input;
}

void Sequential::backward(std::vector<std::vector<float>> error, std::vector<std::vector<float>> inputs)
{
	//std::cout << "errors dim = " << error.size() << " x " << error[0].size() << "\n";
	//std::cout << "inputs dim = " << inputs.size() << " x " << inputs[0].size() << "\n";
	//Finding the deltas for the output layer
	int last_index = layers.size()-1;
	//std::cout << "\nErrors\n";
        //printFloatMatrix(error);
        //std::cout << "\n";

	std::vector<std::vector<float>> current_error = math.matrix_transpose(error);

	//std::cout << "\nCurrent Errors\n";
        //printFloatMatrix(current_error);
        //std::cout << "\n";

	//std::cout << "\nActivations\n";
        //printFloatMatrix(layers[last_index].getActivations());
        //std::cout << "\n";

	//see the src/math_funcs.cpp file for an explanation of how this method works
	//It's basically multiplying element by element in a transposed manner the derivative of the activations by the errors and saving in the deltas.
	std::vector<std::vector<float>> temp_deltas = layers[last_index].getDeltas();
	math.transposed_element_matrix_mult(layers[last_index].getActivations(), current_error, temp_deltas, math.derivative_sigmoid);

	//std::cout << "\nTemp Deltas\n";
        //printFloatMatrix(temp_deltas);
        //std::cout << "\n";

	//doing this cause i was getting a compiling error
	layers[last_index].setDeltas(temp_deltas);

	//get the last layer's weights and deltas
	std::vector<std::vector<float>> current_weights = layers[last_index].getWeights();
	current_error = layers[last_index].getDeltas();

	//computing deltas for all other layers
	//careful with size_t as int leads to unexpected behaviour
	//apparently can't go negative
	for(int i = last_index-1; i >= 0; i--)
	{
		//std::cout << "\nErrors: \n";
		//for(std::vector<float> in : current_error)
                //        printFloatVector(in);
		//std::cout << "delta " << current_error.size() << " delta[0] " << current_error[0].size() << "\n";
		//std::cout << "weights " << current_weights.size() << " weights[0] " << current_weights[0].size() << "\n";
		layers[i].computeDeltas(current_error, current_weights);
		current_error = layers[i].getDeltas();
		current_weights = layers[i].getWeights();
	}

	//updating the weights for all but the first (input) layer
	
	for(size_t i = last_index; i > 0; i--)
	{
		//std::cout << "\nlayer " << i << " weights before update\n";
		//layers[i].printWeights();
		
		//layers[i].updateWeightsLegacy(layers[i-1].getActivations());
		layers[i].updateWeights(layers[i-1].getActivations());
		
		//std::cout << "\nlayer " << i << " weights after update\n";
                //layers[i].printWeights();

	}
	//updating the first layer
	//layers[0].updateWeightsLegacy(inputs);
	layers[0].updateWeights(inputs);
}

void Sequential::trainIteration(std::vector<std::vector<float>> training_inputs, std::vector<std::vector<float>> labels)
{
	std::vector<std::vector<float>> preds = this->forward(training_inputs);

	this->backward(lossFunctionDerivative(preds, labels), training_inputs);
}

std::vector<std::vector<float>> Sequential::lossFunctionDerivative(std::vector<std::vector<float>> predictions, std::vector<std::vector<float>> labels)
{

	//std::cout << "predictions dim = " << predictions.size() << " x " << predictions[0].size() << "\n";
	//std::cout << "labels dim = " << labels.size() << " x " << labels[0].size() << "\n";
	std::vector<std::vector<float>> errors;
	for (size_t i = 0; i < predictions.size(); i++)
	{
		std::vector<float> error;
		for (size_t j = 0; j < predictions[i].size(); j++)
			error.emplace_back(labels[i][j] - predictions[i][j]);
		errors.emplace_back(error);
	}
	//std::cout << "errors dim = " << errors.size() << " x " << errors[0].size() << "\n";
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

