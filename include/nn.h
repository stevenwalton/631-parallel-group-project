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
        size_t num_inputs;
        size_t num_neurons;
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
        void updateWeights(std::vector<float>, std::vector<float>);
        void zeroGrad();
	
        // Helper Classes
        // Printers
	void printActivations();
	void printWeights();
	void printBias();
        void printNodeWeights(struct node);
        // Setters
	void setLearningRate(float);
        // Getters 
	std::vector<float> getActivations();
	std::vector<float> getDeltas();
	std::vector<std::vector<float>> getWeights();
        inline std::vector<struct node> getNeurons() {return neurons;}
	inline float getLearningRate(){return this->learning_rate;};
	inline int getNumInputs(){return this->num_inputs;};
	inline int getNumNeurons(){return this->num_neurons;};
	
};

#endif

