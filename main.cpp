#include <iostream>
#include "nn.h"
#include <vector>
#include <cassert>
#include <time.h>

int main()
{
    //setting a random seed
    srand (time(NULL));

    //toy inputs and label just to make sure everything works
    float inputs[] = {2.0, 3.5};
    float labels[] = {1.0};

    //converting the array into a vector
    //dunno if this is the best way to do it
    //god I miss Python
    std::vector<float> example_input (inputs, inputs + sizeof(inputs) / sizeof(float) );
    std::vector<float> example_labels (labels, labels + sizeof(labels) / sizeof(float) );

    //creating the layers
    //
    LinearLayer inputLayer(2,3);
    LinearLayer l1(3,3);
    LinearLayer l2(3,3);
    LinearLayer outputLayer(3,1);

    std::cout << "Input layer weights:\n\n";
    inputLayer.printWeights();

    std::cout << "\nInput layer deltas:\n\n";
    for (float d : inputLayer.getDeltas())
	    std::cout << d << " ";
    std::cout << "\n";

    /************* forward pass *********************/

    inputLayer.forward(example_input);
    l1.forward(inputLayer.getActivations());
    l2.forward(l1.getActivations());
    outputLayer.forward(l2.getActivations());

    std::cout << "\nOutput: "; 
    outputLayer.printActivations();  
    //Sequential mlp();


    /*************** computing loss function derivative *************/
    std::vector<float> loss_derivative;
    std::vector<float> outputs = outputLayer.getActivations();
    std::vector<std::vector<float>> mock_weights;
    assert(example_labels.size() == outputs.size());

    for(int i = 0; i < outputs.size(); i++)
    {	
	std::vector<float> mock_single_weight;
	mock_single_weight.emplace_back(1.0);
    	mock_weights.emplace_back(mock_single_weight);

	loss_derivative.emplace_back(example_labels[i]-outputs[i]);
    }

    /****************** backprop *************************/
    //First computing the deltas for all the layers
    outputLayer.computeDeltas(loss_derivative, mock_weights);
    l2.computeDeltas(outputLayer.getDeltas(), outputLayer.getWeights());
    l1.computeDeltas(l2.getDeltas(), l2.getWeights());
    inputLayer.computeDeltas(l1.getDeltas(), l1.getWeights());
    
    std::cout << "\nUpdated input layer deltas:\n\n";
    for (float d : inputLayer.getDeltas())
            std::cout << d << " ";
    std::cout << "\n";



    //updating the weights
    outputLayer.updateWeights(l2.getActivations());
    l2.updateWeights(l1.getActivations());
    l1.updateWeights(inputLayer.getActivations());
    inputLayer.updateWeights(example_input);

    std::cout << "\nUpdated input layer weights:\n\n";
    inputLayer.printWeights();
    return 0;
}
