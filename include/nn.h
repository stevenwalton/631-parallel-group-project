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
	//not sure learning rate should be a layer attribute
	//however, I'm setting it to 3 as default just to check that its working
        float learning_rate; 

        std::vector<struct node> neurons;
        //std::vector<struct node> output_nodes;

    public:
        LinearLayer(int, int, float=0.3);
        void initializeLayer();
        // Main Components
        void forward(std::vector<float>);
	void computeDeltas(std::vector<float>, std::vector<std::vector<float>>);
        void updateWeights(std::vector<float>);
	//void backward(std::vector<struct node>); //backward will be a method of sequential
        void zeroGrad();
	
        // Helper Classes
        /*
	 *I don't think this connection functions are needed
	 *
	void connect(std::vector<struct node> input)
                     {std::copy(input_nodes.begin(), 
                      input_nodes.end(), 
                      back_inserter(input));};
        void inputConnect(std::vector<struct node> x)
                     {std::copy(input_nodes.begin(), 
                      input_nodes.end(), 
                      back_inserter(x));};
        */
        std::vector<struct node> getNeurons()
                     {return neurons;}
	
	void printActivations();
	void printWeights();
	void printBias();
        void printNodeWeights(struct node);
	int getNumInputs();
	int getNumNeurons();
	float getLearningRate();
	void setLearningRate(float);
	std::vector<float> getActivations();
	std::vector<float> getDeltas();
	std::vector<std::vector<float>> getWeights();
	
};

#endif

