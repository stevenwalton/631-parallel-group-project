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
        int num_input;
        int num_output;
        int learning_rate = 0;

        std::vector<struct node> input_nodes;
        std::vector<struct node> output_nodes;

    public:
        LinearLayer(int, int);
        void InitializeLayer();
        // Main Components
        void Forward();
        void Backward(std::vector<struct node>, std::vector<struct node>);

        // Helper Classes
        void Connect(std::vector<struct node> input)
                     {std::copy(input_nodes.begin(), 
                      input_nodes.end(), 
                      back_inserter(input));};

        std::vector<struct node> Get_Output()
                     {return output_nodes;}

};

#endif

