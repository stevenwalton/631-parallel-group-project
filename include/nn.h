#ifndef NN_H
#define NN_H

#include <array>
#include <vector>
#include <cmath>
#include "types.h"
#include "math_funcs.h"

class LinearLayer
{
    private:
        math_funcs math;
        int num_inputs;
        int num_outputs;
        int learning_rate = 0;
        float in_bias = 0;
        float in_bias_grad = 0;
        float out_bias = 0;
        float out_bias_grad = 0;

        int num_output_weights = 0;
        int num_input_weights = 0;

        std::vector<struct node> input_nodes;
        std::vector<struct node> output_nodes;

    public:
        LinearLayer(int, int);
        void InitializeLayer();
        // Main Components
        void Forward(std::vector<struct node>, float);
        void Backward(std::vector<struct node>);
        void Zero_Grad();

        // Helper Classes
        void Connect(std::vector<struct node> input)
                     {std::copy(input_nodes.begin(), 
                      input_nodes.end(), 
                      back_inserter(input));};
        void InputConnect(std::vector<struct node> x)
                     {std::copy(input_nodes.begin(), 
                      input_nodes.end(), 
                      back_inserter(x));};
        void SetOutputWeights(std::vector<struct node>);

        std::vector<struct node> Get_Input()
                     {return input_nodes;}
        std::vector<struct node> Get_Output()
                     {return output_nodes;}
        void PrintActivations(std::vector<struct node>);
        void PrintWeights(node);

};

#endif

