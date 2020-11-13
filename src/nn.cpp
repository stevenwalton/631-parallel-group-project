#include <iostream>
#include "nn.h"
#include "types.h"
#include "math_funcs.h"
#include <cassert>

LinearLayer::LinearLayer(int num_inputs, int num_outputs, float lr)
{
    this->num_inputs = num_inputs;
    this->num_neurons = num_outputs;
    this->learning_rate = lr;
    neurons.resize(this->num_neurons);
    //output_nodes.resize(this->num_outputs);
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
        // Initialize then fill
        i.weight.resize(this->num_inputs);
        // No real speedup (unsurprising)
        #pragma omp parallel for
        for (size_t j = 0; j < this->num_inputs; ++j)
            i.weight[j] = math.unit_random();
    }
}

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
     * computed activations instead. Something like:
     *
     * return this.getActivations();
     *
     *
     * I think this is good because we can store everything in the class itself.
     * - Steven
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

/*First step of backprop
 * I'm using independent deltas and weights vectors here instead of passing an actual
node vector because the output layer is computed differently.
This way, for the output layer, we can just pass the derivative of the loss function 
as the delta and a corresponding array of weights set to 1.
For all other layers we pass the deltas and weights of the next layer.
*/
void LinearLayer::computeDeltas(std::vector<float> deltas, std::vector<std::vector<float>> weights) 
{
	//Zero_Grad(); // Kills gradient accumulation, which we aren't doing
	
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
        /*
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
        */
}

/*
 * The complement of the computeDeltas function.
 * This function will update the weights of a layer based on the computed deltas.
 * As such, it must be called after computeDeltas.
 */
void LinearLayer::updateWeights(std::vector<float> input)
{
    // No difference in range based vs iterator based
    for (node& n : this->neurons)
    {
        n.bias += n.delta * this->learning_rate;
        for (size_t i = 0; i < n.weight.size(); i++)
        {
                n.weight[i] += input[i] * n.delta * this->learning_rate;
        }
    }

    /*
    // 3x slower
    //#pragma omp parallel for
    for (auto n = this->neurons.begin(); n < this->neurons.end(); ++n)
    {
        (*n).bias += (*n).delta * this->learning_rate;
        // 10x slower
        //#pragma omp parallel for 
        for (size_t i = 0; i < (*n).weight.size(); ++i)
            (*n).weight[i] += input[i] * (*n).delta * this->learning_rate;

    }
    */
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

inline void LinearLayer::printActivations()
{
    for (node& i : this->neurons)
        std::cout << i.activation << " ";
    std::cout << std::endl;
}

inline void LinearLayer::printBias()
{
    for (node& i : this->neurons)
	    std::cout << i.bias << " ";
    std::cout << std::endl;
}

inline void LinearLayer::printNodeWeights(struct node n)
{
    for (size_t i = 0; i < n.weight.size(); ++i)
        std::cout << n.weight[i] << " ";
    std:: cout << std::endl;
}

inline void LinearLayer::printWeights()
{
	for (node& i : this->neurons)
		this->printNodeWeights(i);
}
