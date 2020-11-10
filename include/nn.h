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
        int num_neurons;
        int learning_rate = 0;
        //float bias = 0;
        //float bias_grad = 0;
        //float out_bias = 0;
        //float out_bias_grad = 0;

        //int num_weights = 0;
        //int num_input_weights = 0;

        std::vector<struct node> neurons;
        //std::vector<struct node> output_nodes;

    public:
        LinearLayer(int, int);
        void initializeLayer();
        // Main Components
        void forward(std::vector<float>);
        void backward(std::vector<struct node>);
        void zeroGrad();

        // Helper Classes
        /*
	void connect(std::vector<struct node> input)
                     {std::copy(input_nodes.begin(), 
                      input_nodes.end(), 
                      back_inserter(input));};
        void inputConnect(std::vector<struct node> x)
                     {std::copy(input_nodes.begin(), 
                      input_nodes.end(), 
                      back_inserter(x));};
        void SetOutputWeights(std::vector<struct node>);
        
        std::vector<struct node> Get_Input()
                     {return input_nodes;}
        std::vector<struct node> Get_Output()
                     {return output_nodes;}
        */
        std::vector<struct node> getNeurons()
                     {return neurons;}
	
	void printActivations();
	void printWeights();
	void printBias();
        void printNodeWeights(struct node);
	std::vector<float> getActivations();

};

#endif

