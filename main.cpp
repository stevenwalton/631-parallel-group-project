#include <iostream>
#include "nn.h"
#include "sequential.h"
#include <vector>
#include <cassert>
#include <time.h>

int main()
{
    //setting a random seed
    srand (time(NULL));

    //toy inputs and labels
    std::vector<float> example_input{2.0, 3.5};
    std::vector<float> example_labels{1.0};

    //creating the layers
    //
    LinearLayer inputLayer(2,3);
    LinearLayer l1(3,3);
    LinearLayer l2(3,3);
    LinearLayer outputLayer(3,1);

    Sequential model;
    //Adding the layers to the model
    model.add(inputLayer);
    model.add(l1);
    model.add(l2);
    model.add(outputLayer);

    model.printModel();

    std::cout << "Input layer weights:\n\n";
    model.getLayers()[0].printWeights();

    std::cout << "\nInput layer deltas:\n\n";
    for (float d : model.getLayers()[0].getDeltas())
	    std::cout << d << " ";
    std::cout << "\n";

    /************* train one sample *********************/

    model.trainIteration(example_input, example_labels);

    //std::cout << "\nOutput: "; 
    //outputLayer.printActivations();  

    std::cout << "\nUpdated input layer deltas:\n\n";
    for (float d : model.getLayers()[0].getDeltas())
            std::cout << d << " ";
    std::cout << "\n";

    std::cout << "\nUpdated input layer weights:\n\n";
    model.getLayers()[0].printWeights();
    
    return 0;
}
