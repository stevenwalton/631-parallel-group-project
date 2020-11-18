#include <iostream>
#include "nn.h"
#include "types.h"
#include "math_funcs.h"
#include "sequential.h"
#include <vector>
		
Sequential::Sequential(){
}

std::vector<float> Sequential::forward(std::vector<float> inputs)
{
	std::vector<float> current_input(inputs.begin(), inputs.end());
	for (size_t i = 0; i < layers.size(); ++i)
	{
		layers[i].forward(current_input);
		current_input = layers[i].getActivations();
	}
	//returning the last layer's activations
	return current_input;
}

std::vector<std::vector<float> > Sequential::batchForward(std::vector<std::vector<float> > batch_inputs)
{
    std::vector<std::vector<float> > batch_preds(this->batch_size);
    for (size_t i = 0; i < this->batch_size; ++i)
        batch_preds[i] = (this->forward(batch_inputs[i]));
    return batch_preds;

}

void Sequential::batchBackward(std::vector<std::vector<float> > batch_preds,
                               std::vector<std::vector<float> > batch_labels,
                               std::vector<std::vector<float> > batch_inputs)
{
    size_t s = (batch_preds[0]).size(); // Size of our data
    std::vector<float> mb_mean(s,0);
    std::vector<float> mb_var(s,0);
    std::vector<std::vector<float> > mb_norm;
    double epsilon = 0.00001;
    for (size_t i = 0; i < this->batch_size; ++i)
        mb_norm.emplace_back(mb_mean);

    for (size_t i = 0; i < this->batch_size; ++i)
        for (size_t j = 0; j < s; ++j)
            mb_mean[j] += batch_preds[i][j];

    for (size_t i = 0; i < s; ++i)
        mb_mean[i] /= this->batch_size;

    for (size_t i = 0; i < this->batch_size; ++i)
        for (size_t j = 0; j < s; ++j)
            mb_var[j] += pow(batch_preds[i][j] - mb_mean[j], 2.);

    // epsilon = 0
    for (size_t i = 0; i < s; ++i)
        mb_var[i] = mb_var[i] / this->batch_size;

    for (size_t i = 0; i < this->batch_size; ++i)
        for (size_t j = 0; j < s; ++j)
            mb_norm[i][j] = (batch_preds[i][j] - mb_mean[j]) / sqrt(mb_var[j] + epsilon);

    for (size_t i = 0; i < this->batch_size; ++i)
    {
        std::vector<float> lfd = lossFunctionDerivative(mb_norm[i], batch_labels[i]);
        this->backward(lfd, batch_inputs[i]);
    }

    /*
    size_t s = (batch_preds[0]).size();
    std::cout << batch_preds.size() << " " << batch_preds[0].size() << std::endl;
    std::vector<float> mean_der(s, 0); 
    std::vector<float> mean_inputs(s, 0); 
    #pragma omp parallel for
    for (size_t i = 0; i < this->batch_size; ++i)
    {
        for (size_t j = 0; j < s; ++j)
            mean_inputs[j] += batch_inputs[i][j];
    }
    #pragma omp parallel for
    for (size_t i = 0; i < s; ++i)
        mean_inputs[i] /= this->batch_size;

    #pragma omp parallel for
    for (size_t i = 0; i < this->batch_size; ++i)
    {
        std::vector<float> lfd = lossFunctionDerivative(batch_preds[i], batch_labels[i]);
         for (size_t j = 0; j < s; ++j)
             mean_der[j] += lfd[j];
    }
    #pragma omp parallel for
    for (size_t i = 0; i < s; ++i)
        mean_der[i] /= this->batch_size;
    //printf("Mean Der[0] %f\t", mean_der[0]);
    //printf("Mean in[0] %f\n", mean_inputs[0]);
    this->backward(mean_der, mean_inputs);
    */
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
	std::vector<float> preds = this->forward(training_inputs);
	
	this->backward(lossFunctionDerivative(preds, labels), training_inputs);
}

void Sequential::batchTrainIteration(std::vector<std::vector<float> > batch_training_inputs,
                                     std::vector<std::vector<float> > batch_labels)
{
    std::vector<std::vector<float> > batch_preds = this->batchForward(batch_training_inputs);

    //std::vector<std::vector<float> > loss;
    //for (size_t i = 0; i < this->batch_size; ++i)
    //    loss.emplace_back(batch_preds[i], batch_labels[i]);
    this->batchBackward(batch_preds, batch_labels, batch_training_inputs);
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

