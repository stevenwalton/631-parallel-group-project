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

/***************** Helper functions **********************************/
void Sequential::printModel()
{
    std::cout << "Model has " << this->layers.size() << " layers: \n";
    for (size_t i = 0; i < layers.size(); ++i)
        std::cout << "Layer "<< i << ": (" << layers[i].getNumInputs() << ", " << layers[i].getNumNeurons() << ")\n";
    std::cout << "\n";

}

