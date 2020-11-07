#include <iostream>
#include "nn.h"
#include "types.h"

LinearLayer::LinearLayer(int num_inputs, int num_outputs)
{
    this->num_inputs = num_inputs;
    this->num_outputs = num_outputs;
    input_nodes.resize(this->num_inputs);
    output_nodes.resize(this->num_outputs);
    this->num_input_weights = num_outputs;
    InitializeLayer();
}

void LinearLayer::InitializeLayer()
{
    // TODO: Change to random
    //       Find fast random
    for (node& i : this->input_nodes)
    {
        i.activation = 0;
        i.error = 0;
        // 
        for (size_t j = 0; j < this->num_inputs; ++j)
            i.weight.emplace_back(1);
    }
    for (node& i : output_nodes)
    {
        i.activation = 0;
        i.error = 0;
        // We don't resize weights because we don't know how many
    }
}

void LinearLayer::SetOutputWeights(std::vector<struct node> connection)
{
    int num_nodes = connection.size();
    for (node& i : output_nodes)
        i.weight.emplace_back(1);
}

void LinearLayer::Zero_Grad()
{
    /*
     * Sets the gradients to zero
     */
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
    }
}

void LinearLayer::Forward(std::vector<struct node> x, float x_bias)
{
    /*
     * Performs the feed forward section of the network
     * activation = weight * input + bias
     */
    // Input Layer
    size_t ii = 0;
    for (node& i : input_nodes)
    {
        ii = 0;
        i.activation = 0;
        for (node& j : x)
        {
            i.activation += (j.weight[ii] * j.activation);
            ii++;
        }
        i.activation = math.sigmoid(i.activation + x_bias);
    }
    // Output Layer
    for (node& i : output_nodes)
    {
        ii = 0;
        i.activation = 0;
        for (node& j : input_nodes)
        {
            i.activation += (j.weight[ii] * j.activation);
            ii++;
        }
        i.activation = math.sigmoid(i.activation + in_bias);
    }
}

void LinearLayer::Backward(std::vector<struct node> y) 
{
    // Output Layer
    Zero_Grad(); // Kills gradient accumulation, which we aren't doing
    math.MSE(output_nodes, y);
    size_t ii = 0;
    // TODO
    for (node& i : output_nodes)
    {
        for (size_t j = 0; j < this->num_output_weights; ++j)
        {
        }
    }
}

/******************** Helper Functions ********************/  
void LinearLayer::PrintActivations(std::vector<struct node> n)
{
    for (node& i : n)
        std::cout << i.activation << " ";
    std::cout << std::endl;
}

void LinearLayer::PrintWeights(node n)
{
    for (size_t i = 0; i < n.weight.size(); ++i)
        std::cout << n.weight[i] << " ";
    std:: cout << std::endl;
}
