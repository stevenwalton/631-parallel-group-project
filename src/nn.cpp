#include "nn.h"
#include "types.h"

LinearLayer::LinearLayer(int num_inputs, int num_outputs)
{
    input_nodes.resize(num_inputs);
    input_nodes.resize(num_outputs);
}

void LinearLayer::InitializeLayer()
{
    // TODO: Change to random
    //       Find fast random
    for (node& i : input_nodes)
    {
        i.activation = 0;
        i.weight = 0;
        i.bias= 0;
    }
    for (node& i : output_nodes)
    {
        i.activation = 0;
        i.weight = 0;
        i.bias= 0;
    }
}

void LinearLayer::Forward()
{
    for (node& i : output_nodes)
    {
        for (node& j : input_nodes)
        {
            i.activation += (j.weight * j.activation) + j.bias;
        }
    }
}

void LinearLayer::Backward()
{
}

void LinearLayer::Connect(std::vector<struct node> input)
{
    std::copy(input_nodes.begin(), input_nodes.end(), back_inserter(input));
}
