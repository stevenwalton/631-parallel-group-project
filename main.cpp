#include <iostream>
#include "nn.h"
#include <vector>

int main()
{
    float inputs[] = {2.0, 3.5};
    std::vector<float> example_input (inputs, inputs + sizeof(inputs) / sizeof(float) );
    
    LinearLayer inputLayer(2,3);
    LinearLayer l1(3,3);
    LinearLayer l2(3,3);
    LinearLayer outputLayer(3,1);

    inputLayer.forward(example_input);
    l1.forward(inputLayer.getActivations());
    l2.forward(l1.getActivations());
    outputLayer.forward(l2.getActivations());

    outputLayer.printActivations();  
    //Sequential mlp();

    return 0;
}
