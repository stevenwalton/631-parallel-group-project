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

	for (size_t i = 0; i < layers.size(); ++i)
	{
		//std::cout << "ci size = " << current_input.size() << "\n";
        	//std::cout << "ci[0] size = " << current_input[0].size() << "\n";

		//std::cout << "\n Layer " << i << " inputs \n\n";

		//for(std::vector<float> in : current_input)
                //        printFloatVector(in);

		
		std::cout << "\nLayer " << i << " weights \n";
		layers[i].printWeights();

		//std::cout << "\n Layer " << i << " bias \n\n";
		//layers[i].printBias();

		layers[i].forward(current_input);
		current_input = layers[i].getActivations();
	}
	//returning the last layer's activations
	//for(std::vector<float> in : current_input)
        //                printFloatVector(in);
	return current_input;
}

void Sequential::backward(std::vector<std::vector<float>> error, std::vector<std::vector<float>> inputs)
{
	//std::cout << "errors dim = " << error.size() << " x " << error[0].size() << "\n";
	//std::cout << "inputs dim = " << inputs.size() << " x " << inputs[0].size() << "\n";
	//Creating the fake 'weights' vector of vectors for the 
	//output layer
	std::vector<std::vector<float>> current_weights;
	//for(float f : error)
        for (size_t i = error[0].size(); i != 0; --i) 
	{
		std::vector<float> w{1.0};
		current_weights.emplace_back(w);
	}
	//computing deltas
	std::vector<std::vector<float>> current_error(error.begin(), error.end());
	//careful with size_t as int leads to unexpected behaviour
	//apparently can't go negative
	for(int i = layers.size()-1; i >= 0; i--)
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
	
	for(size_t i = layers.size()-1; i > 0; i--)
	{
		//std::cout << "\nlayer " << i << " weights before update\n";
		//layers[i].printWeights();
		layers[i].updateWeightsLegacy(layers[i-1].getActivations());
		//std::cout << "\nlayer " << i << " weights after update\n";
                //layers[i].printWeights();

	}
	//updating the first layer
	layers[0].updateWeightsLegacy(inputs);
	
}

void Sequential::trainIteration(std::vector<std::vector<float>> training_inputs, std::vector<std::vector<float>> labels)
{
	std::cout << "\nTraining Inputs: \n";
        for(std::vector<float> in : training_inputs)
                        printFloatVector(in);
	std::vector<std::vector<float>> preds = this->forward(training_inputs);


	std::cout << "\nPredictions: \n";
	for(std::vector<float> in : preds)
                        printFloatVector(in);

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

