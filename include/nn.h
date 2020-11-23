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
        float learning_rate; 
	size_t batch_size;
        std::vector<struct node> neurons;

    public:
        LinearLayer(int, int, float=0.3, int=32);
        void initializeLayer();
        // Main Components
        void forward(std::vector<std::vector<float>>);
	void computeDeltas(std::vector<std::vector<float>>, std::vector<std::vector<float>>);
        void updateWeights(std::vector<std::vector<float>>);
        void updateWeightsLegacy(std::vector<std::vector<float>>);
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
	std::vector<std::vector<float>> getActivations();
	std::vector<std::vector<float>> getDeltas();
	std::vector<std::vector<float>> getWeights();
        inline std::vector<struct node> getNeurons() {return neurons;}
	inline float getLearningRate(){return this->learning_rate;};
	inline int getNumInputs(){return this->num_inputs;};
	inline int getNumNeurons(){return this->num_neurons;};
	
};

#endif

