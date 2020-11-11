#include <iostream>
#include "nn.h"
#include "types.h"
#include "math_funcs.h"
#include <cassert>

LinearLayer::LinearLayer(int num_inputs, int num_outputs)
{
    this->num_inputs = num_inputs;
    this->num_neurons = num_outputs;
    neurons.resize(this->num_neurons);
    //output_nodes.resize(this->num_outputs);
    //this->num_weights = num_inputs;
    initializeLayer();
}

void LinearLayer::initializeLayer()
{
    // TODO: Find faster random
    for (node& i : this->neurons)
    {
        i.activation = 0;
        i.error = 0;
	i.bias = math.unit_random();
        // 
        for (size_t j = 0; j < this->num_inputs; ++j)
            i.weight.emplace_back(math.unit_random());
    }
}

/*void LinearLayer::SetOutputWeights(std::vector<struct node> connection)
{
    int num_nodes = connection.size();
    for (node& i : output_nodes)
        i.weight.emplace_back(1);
}*/

void LinearLayer::zeroGrad()
{
    /*
     * Sets the gradients to zero
     
    // input layer
    this->in_bias_grad = 0;
    for (node& i : this->input_nodes)
    {
        for (size_t j = 0; j < this->num_inputs; ++j)
            i.weight_grad[j] = 0;
    }
    // output layer
    this->out_bias_grad = 0;
    for (node& i : this->output_nodes)
    {
        for (size_t j = 0; j < this->num_outputs; ++j)
            i.weight_grad[j] = 0;
    }*/
}

void LinearLayer::forward(std::vector<float> input)
{
    /*
     * Performs the feed forward section of the network
     * activation = weight * input + bias
     *
     *
     * Maybe this should not be void and return the 
     * computed activations instead. Somehting like:
     *
     * return this.getActivations();
     */
    //std::cout << input.size() << std::endl;
    //std::cout << this->num_inputs << std::endl; 
    assert(input.size() == this->num_inputs);
    size_t ii = 0;
    for (node& i : this->neurons)
    {
        ii = 0;
        i.activation = 0;
        for (float act : input)
        {
            i.activation += (i.weight[ii] * act);
            ii++;
        }
        i.activation = math.sigmoid(i.activation + i.bias);
    }
}

//I'm using independent deltas and weights vectors here instead of passing an actual
//node vector because the output layer is computed differently
//this way we can just pass the derivative of the loss function as the delta and a
//corresponding array of weights just set to 1
void LinearLayer::computeDeltas(std::vector<float> deltas, std::vector<std::vector<float>> weights) 
{
	//Zero_Grad(); // Kills gradient accumulation, which we aren't doing
	
	//on an inner node, that is, not the output node
	size_t jj;
	size_t ii = 0;
	for (node& i : this->neurons)
	{
		jj = 0;
		i.error = 0.0;
		for (float d : deltas)
		{
			i.error += d * weights[ii][jj];
			jj++;
		}
		i.delta = i.error * math.derivative_sigmoid(i.activation);
	}
}

/*
 * The complement of the computeDeltas function.
 * This function will update the weights of a layer based on the computed deltas.
 * As such, it must be called after computeDeltas.
 */
void LinearLayer::updateWeights(std::vector<float> input)
{
	for (node& n : this->neurons)
	{
		n.bias += n.delta * this->learning_rate;
		for (size_t i = 0; i < n.weight.size(); i++)
		{
			n.weight[i] += input[i] * n.delta * this->learning_rate;
		}
	}
}

/******************** Helper Functions ********************/
std::vector<float> LinearLayer::getActivations()
{
	std::vector<float> activations;
	for (node& n: this->neurons){
            activations.emplace_back(n.activation);
	}
	return activations;
}

std::vector<float> LinearLayer::getDeltas()
{
        std::vector<float> deltas;
        for (node& n: this->neurons){
            deltas.emplace_back(n.delta);
        }
        return deltas;
}

std::vector<std::vector<float>> LinearLayer::getWeights()
{
        std::vector<std::vector<float>> weights;
        for (node& n: this->neurons){
            weights.emplace_back(n.weight);
        }
        return weights;
}

//vector<vector<data_type>> vec;

void LinearLayer::printActivations()
{
    for (node& i : this->neurons)
        std::cout << i.activation << " ";
    std::cout << std::endl;
}

void LinearLayer::printBias()
{
    for (node& i : this->neurons)
	    std::cout << i.bias << " ";
    std::cout << std::endl;
}

void LinearLayer::printNodeWeights(struct node n)
{
    for (size_t i = 0; i < n.weight.size(); ++i)
        std::cout << n.weight[i] << " ";
    std:: cout << std::endl;
}

void LinearLayer::printWeights()
{
	for (node& i : this->neurons)
		this->printNodeWeights(i);
}
