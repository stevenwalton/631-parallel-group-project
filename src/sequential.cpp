#include <iostream>
#include "nn.h"
#include "types.h"
#include "math_funcs.h"
#include "sequential.h"
#include "utils.h"
#include <vector>
#include "utils.h"

Sequential::Sequential(){
}

std::vector<std::vector<float>> Sequential::forward(std::vector<std::vector<float>> inputs)
{
    std::vector<std::vector<float>> current_input(inputs.begin(), inputs.end());

    for (size_t i = 0; i < layers.size(); ++i)
    {
            layers[i].forward(current_input);
            current_input = layers[i].getActivations();
    }
    return current_input;
}

void Sequential::backward(std::vector<std::vector<float>> error, std::vector<std::vector<float>> inputs)
{
    //Finding the deltas for the output layer
    int last_index = layers.size()-1;
    std::vector<std::vector<float>> current_error = math.matrix_transpose(error);

    //see the src/math_funcs.cpp file for an explanation of how this method works
    //It's basically multiplying element by element in a transposed manner the derivative of the activations by the errors and saving in the deltas.
    std::vector<std::vector<float>> temp_deltas = layers[last_index].getDeltas();
    math.transposed_element_matrix_mult(layers[last_index].getActivations(), current_error, temp_deltas, math.derivative_sigmoid);

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
        layers[i].computeDeltas(current_error, current_weights);
        current_error = layers[i].getDeltas();
        current_weights = layers[i].getWeights();
    }

    //updating the weights for all but the first (input) layer
    
    for(size_t i = last_index; i > 0; i--)
        layers[i].updateWeights(layers[i-1].getActivations());
    
    //updating the first layer
    layers[0].updateWeights(inputs);
}

void Sequential::trainIteration(std::vector<std::vector<float>> training_inputs, std::vector<int> labels)
{
    std::vector<std::vector<float>> preds = this->forward(training_inputs);
    this->backward(lossFunctionDerivative(preds, labels), training_inputs);
    //this->backward(crossEntropyLossDerivative(preds, labels), training_inputs);
}

std::vector<std::vector<float>> Sequential::lossFunctionDerivative(std::vector<std::vector<float>> predictions, std::vector<int> labels)
{

    size_t pred_size = predictions.size();
    std::vector<std::vector<float>> errors(pred_size);
    for (size_t i = 0; i < pred_size; i++)
    {
        size_t predi_size = predictions[i].size();
        std::vector<float> error(predi_size);
        for (size_t j = 0; j < predi_size; j++)
            error[j] = predictions[i][j] - labels[i];
        errors[i] = error;
    }
    return errors;
}

std::vector<std::vector<float>> Sequential::crossEntropyLossDerivative(std::vector<std::vector<float>> logits, std::vector<int> labels)
{
        //logits has size <batch_size, num_classes>
        //labels has size <batch_size> and each input is in the range [0, num_classes-1]
	
	//creating a copy of the logits
	std::vector<std::vector<float>> softmax(logits);
	
	//taking the exp of the logits
	math.map_function(softmax, exp);

	//std::cout << "\nExp logits: \n";
        //printFloatMatrix(softmax);

	//summing the log exp logits for all classes
	std::vector<float> sums = math.sumRows(softmax);

	//std::cout << "\nSummed rows: \n";
        //for(float in : sums)
        //        std::cout << in << " ";
        //std::cout << std::endl;

	//We need to do softmax / sums effectively dividing each element in softmax by 
	//its corresponding sum to get a vector that sums to 1
	math.inverse_scale_matrix(sums, softmax);
	
	//std::cout << "\nSoftmax: \n";
        //printFloatMatrix(softmax);
	
	//creating a zeros matrix like logits
	std::vector<std::vector<float>> correct_classes(logits.size(), std::vector<float>(logits[0].size(), 0.0));
	//for(size_t i = 0; i < logits.size(); ++i):
	//	correct_classes.push_back(std::vector<float>(logits[0].size(), 0.0));

	//Setting 1.0 in the index of the correct class
	math.setMatrix2Value(correct_classes, labels, 1.0);

        //std::cout << "\nCorrect classes: \n";
        //printFloatMatrix(correct_classes);
	
	//doing (-correct_classes + softmax)
	math.matrix_add(softmax, correct_classes, softmax);

        //std::cout << "\nSoftmax-correct_classes: \n";
        //printFloatMatrix(softmax);

	//dividing by batch_size
	math.inverse_scale_matrix(logits.size(), softmax);

	//std::cout << "\nDivided by batch_size: \n";
        //printFloatMatrix(softmax);

	//returning the gradients <batch_size, num_classes>
	return softmax;
}

/***************** Helper functions **********************************/
void Sequential::printModel()
{
    std::cout << "Model has " << this->layers.size() << " layers: \n";
    for (size_t i = 0; i < layers.size(); ++i)
        std::cout << "Layer "<< i << ": (" << layers[i].getNumInputs() << ", " << layers[i].getNumNeurons() << ")\n";
    std::cout << "\n";

}


