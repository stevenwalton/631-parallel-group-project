#include <iostream>
#include "nn.h"
#include "types.h"
#include "utils.h"
#include "math_funcs.h"
#include <cassert>

using namespace std;

LinearLayer::LinearLayer(int num_inputs, int num_outputs, float lr, int batch_size)
{
    this->num_inputs = num_inputs;
    this->num_outputs = num_outputs;
    this->learning_rate = lr;
    this->batch_size = batch_size;
    //weights.resize(this->num_inputs);
    initializeLayer();
}

void LinearLayer::initializeLayer()
{
    int i, j;

    //initializing the weights
    for (i = 0; i < this->num_inputs; i++)
    {
	vector<float> w;
	for(j = 0; j < this->num_outputs; j++)
	{
		w.push_back(math.unit_random());
	}
	this->weights.push_back(w);
    }

    //initializing the bias for each neuron
    for (i = 0; i < this->num_outputs; i++)
    {
        this->bias.push_back(math.unit_random());
    	// deltas and errors are going to be transposed, so n_outputs x batch_size
    	this->deltas.push_back(vector<float>(this->batch_size, 0.0));
        this->errors.push_back(vector<float>(this->batch_size, 0.0));
    }

    for (i = 0; i < this->batch_size; i++)
    {
	this->activations.push_back(vector<float>(this->num_outputs, 0.0));
	//this->deltas.push_back(vector<float>(this->n_outputs, 0.0));
	//this->errors.push_back(vector<float>(this->n_outputs, 0.0));
    }
}

void LinearLayer::zeroGrad()
{
    /*
     * Sets the gradients to zero
     */
	int i, j;
	for (i = 0; i < this->num_outputs; i++)
    	{
		for(j = 0; j < this->batch_size; j++)
		{
			errors[i][j] = 0.0;
			deltas[i][j] = 0.0;
		}
    	}    
}

void LinearLayer::forward(std::vector<std::vector<float>> batch_inputs)
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
    //std::cout << batch_inputs[0].size() << std::endl;
    //std::cout << this->num_inputs << std::endl; 
    
    //computing forward, 
    //this can probably be done more efficiently by combining function behaviors
    //specially for CUDA where if we keep it as 3 distinct operations we would
    //need to allocate memory 3 times for pretty much the same data
	
	//this->printActivations();
	math.matrix_mult(batch_inputs, this->weights, this->activations);
   
	//this->printActivations();
	//std::cout << "\n";

	math.matrix_plus_vec(this->activations, this->bias);
	
	//this->printActivations();
        //std::cout << "\n";
	
	math.map_function(this->activations, math.sigmoid);

	//this->printActivations();
        //std::cout << "\n";
}

/*First step of backprop
 * I'm using independent deltas and weights vectors here instead of passing an actual
node vector because the output layer is computed differently.
This way, for the output layer, we can just pass the derivative of the loss function 
as the delta and a corresponding array of weights set to 1.
For all other layers we pass the deltas and weights of the next layer.
*/
void LinearLayer::computeDeltas(std::vector<std::vector<float>> previous_errors, std::vector<std::vector<float>> weights) 
{
	zeroGrad(); // Set error and deltas to zero
	//**************************
	// 3 x 2        2 x 32         3 x 32
	//weights * previous_errors -> errors
	//std::cout << "Previous errors\n"; 
	//printFloatMatrix(previous_errors);
	//std::cout << "\n";

	math.matrix_mult(weights, previous_errors, this->errors);
	//3 x 32              32 x 3       3 x 32
	//errors *tran_elem deriv(activation) -> deltas
	math.transposed_element_matrix_mult(this->activations, this->errors, this->deltas, math.derivative_sigmoid);
}

/*
 * The complement of the computeDeltas function.
 * This function will update the weights of a layer based on the computed deltas.
 * As such, it must be called after computeDeltas.
 */
void LinearLayer::updateWeights(std::vector<std::vector<float>> inputs)
{

	//*******************************************
	// Basic idea
	// inputs 32 x 3   deltas 2 x 32  weights 3 x 2
	// inputs' x deltas' -> weight_updates
	//  3 x 32 * 32 x 2 -> 3 x 2
	//  3 x 2          3 x 2
	// weights += weight_updates + learning_rate
	//******************************************
	//
	//updating the bias
        for (int i = 0; i < this->bias.size(); i++)
        {
                this->bias[i] -= math.vector_mean(this->deltas[i]) * this->learning_rate;
        }

	//creating a matrix to hold the weight updates
	vector<vector<float>> weight_updates(this->num_inputs, vector<float>(this->num_outputs, 0.0));
	vector<vector<float>> l2_updates(this->weights);

	//for (int i = 0; i < this->num_inputs; i++)
	//	weight_updates.push_back(vector<float>(this->num_outputs, 0.0));

	math.matrix_mult(math.matrix_transpose(inputs), math.matrix_transpose(this->deltas), weight_updates);
	//using the matrix_add method to basically do
	//weights[i][j] = 1.0 * weights[i][j] + lr * weight_updates[i][j]

	math.matrix_add(1.0, this->weights, -1.0 * this->learning_rate, weight_updates, this->weights);

	//l2 reg
	

}

void LinearLayer::updateWeightsLegacy(std::vector<std::vector<float>> inputs)
{
    /*
    float delta_sum, input_delta_sum;
    size_t j, i;
    for (node& n : this->neurons)
    {
	for(j = 0; j < inputs.size(); j++)
        {
		//std::cout << "\ndelta = " << n.delta[j] << "\n";
        	n.bias += n.delta[j] * this->learning_rate;
		for (i = 0; i < n.weight.size(); i++)
        	{
			//std::cout << inputs[j][i] << " * " << n.delta[j] << " * " << this->learning_rate << "\n";
			n.weight[i] += inputs[j][i] * n.delta[j] * this->learning_rate;
		}
	}

    }*/

}


/******************** Helper Functions ********************/
void LinearLayer::printActivations()
{
    	for (size_t i = 0; i < this->batch_size; ++i)
	{
		for(float f : this->activations[i])
        		std::cout << f << " ";
    		std::cout << std::endl;
	}
}

void LinearLayer::printBias()
{
    for (float f : this->bias)
	    std::cout << f << " ";
    std::cout << std::endl;
}

void LinearLayer::printWeights()
{
	for (vector<float> v : this->weights)
	{
		for (float f : v)
			std::cout << f << " ";
		std::cout << std::endl;
	}	
}
