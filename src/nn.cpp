#include <iostream>
#include "nn.h"
#include "types.h"
#include "math_funcs.h"
#include <cassert>

LinearLayer::LinearLayer(int num_inputs, int num_outputs, float lr, int batch_size)
{
    this->num_inputs = num_inputs;
    this->num_neurons = num_outputs;
    this->learning_rate = lr;
    this->batch_size = batch_size;
    neurons.resize(this->num_neurons);
    initializeLayer();
}

void LinearLayer::initializeLayer()
{
    // TODO: Find faster random
    for (node& n : this->neurons)
    {
        n.activation.resize(this->batch_size);
        n.error.resize(this->batch_size);
	n.delta.resize(this->batch_size);
        #pragma omp parallel for schedule(static)
	for (size_t j = 0; j < this->batch_size; ++j){
		n.activation[j] = 0.0;
		n.error[j] = 0.0;
		n.delta[j] = 0.0;
	}
	n.bias = math.unit_random();
        // Initialize then fill
        n.weight.resize(this->num_inputs);
        // No real speedup (unsurprising)
        #pragma omp parallel for
        for (size_t j = 0; j < this->num_inputs; ++j)
            n.weight[j] = math.unit_random();
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
    
    assert(batch_inputs[0].size() == this->num_inputs);
    //iterating over all training instances in the batch
    for (size_t i = 0; i < batch_inputs.size(); i++){
    	for (node& n : this->neurons)
    	{	
        	//j = 0;
        	n.activation[i] = math.dot_product(n.weight, batch_inputs[i]);
        	//for (float act : batch_inputs[i])
        	//{
            	//	n.activation[i] += (n.weight[j] * act);
            	//	j++;
		//}	
        	n.activation[i] = math.sigmoid(n.activation[i] + n.bias);
    	}
    }
}

/*First step of backprop
 * I'm using independent deltas and weights vectors here instead of passing an actual
node vector because the output layer is computed differently.
This way, for the output layer, we can just pass the derivative of the loss function 
as the delta and a corresponding array of weights set to 1.
For all other layers we pass the deltas and weights of the next layer.
*/
void LinearLayer::computeDeltas(std::vector<std::vector<float>> deltas, std::vector<std::vector<float>> weights) 
{
	//Zero_Grad(); // Kills gradient accumulation, which we aren't doing
	
	size_t ii, jj, i;
	//computing for each training instance
	for (i = 0; i < deltas.size(); i++){
		//ii tracks the number of neurons in current layer
		ii = 0;
		//std::cout << "i = " << i << "\n";
		for (node& n : this->neurons)
		{
			//jj tracks number of rows in weights
			//that is number of neurons in layer where weights came from
			jj = 0;

			//std::cout << "ii = " << ii << "\n";
			//std::cout << "delta size = " << deltas[i].size() << "\n";
			//std::cout << "weights size = " << weights.size() << "\n";
			//n.error[ii] = math.dot_product(deltas[i], weights[ii]);
			n.error[i] = 0.0;
			for (float d : deltas[i])
			{
				n.error[i] += d * weights[jj][ii];
				jj++;
			}
			n.delta[i] = n.error[i] * math.derivative_sigmoid(n.activation[i]);
			ii++;
		}
	}
}

/*
 * The complement of the computeDeltas function.
 * This function will update the weights of a layer based on the computed deltas.
 * As such, it must be called after computeDeltas.
 */
void LinearLayer::updateWeights(std::vector<std::vector<float>> inputs)
{

    float delta_sum, input_delta_sum;
    size_t j, i;
    for (node& n : this->neurons)
    {
	delta_sum = math.vector_sum(n.delta);
        n.bias += delta_sum * this->learning_rate;
	input_delta_sum = 0.0;
	for(j = 0; j < inputs.size(); j++)
	{
		//multiplying each input by its corresponding delta and summing everything
		input_delta_sum += math.vector_sum(inputs[j]) * n.delta[j];
	}

	for (i = 0; i < n.weight.size(); i++)
        {
		//updating each weight only once per batch
                n.weight[i] += input_delta_sum * this->learning_rate;
        }
    }

}

void LinearLayer::updateWeightsLegacy(std::vector<std::vector<float>> inputs)
{

    //float delta_sum, input_delta_sum;
    //size_t j, i;
    for (node& n : this->neurons)
    {
	for(size_t j = 0; j < inputs.size(); ++j)
        {
        	n.bias += n.delta[j] * this->learning_rate;
		for (size_t i = 0; i < n.weight.size(); ++i)
        	{
			//std::cout << inputs[j][i] << " * " << n.delta[j] << " * " << this->learning_rate << "\n";
			n.weight[i] += inputs[j][i] * n.delta[j] * this->learning_rate;
		}
	}

    }

}


/******************** Helper Functions ********************/
std::vector<std::vector<float>> LinearLayer::getActivations()
{
	std::vector<std::vector<float>> activations;
	/*for (node& n: this->neurons){
            activations.emplace_back(n.activation);
	}
	return activations;
	*/
	for (size_t i = 0; i < this->batch_size; i++){
		std::vector<float> current;
		for (node& n : this->neurons)
			current.emplace_back(n.activation[i]);
		activations.emplace_back(current);
	}
	//std::cout << "act size = " << activations.size() << "\n";
	return activations;
}

std::vector<std::vector<float>> LinearLayer::getDeltas()
{
	std::vector<std::vector<float>> deltas;
        /*for (node& n: this->neurons){
            deltas.emplace_back(n.delta);
        }*/
	for (size_t i = 0; i < this->batch_size; i++){
                std::vector<float> current;
                for (node& n : this->neurons)
                        current.emplace_back(n.delta[i]);
                deltas.emplace_back(current);
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

void LinearLayer::printActivations()
{
    	for (node& i : this->neurons)
	{
		for(float f : i.activation)
        		std::cout << f << " ";
    		std::cout << std::endl;
	}
}

void LinearLayer::printBias()
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

void LinearLayer::printWeights()
{
	for (node& i : this->neurons)
		this->printNodeWeights(i);
}
