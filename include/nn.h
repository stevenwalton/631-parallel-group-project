#ifndef NN_H
#define NN_H

#include <array>
#include <vector>
#include <cmath>
#include "types.h"
#include "math_funcs.h"

using namespace std;

class LinearLayer
{
    private:
        math_funcs math;
        int num_inputs;
        int num_outputs;
        float learning_rate; 
	int batch_size;
	
	//now I'll store the weights as a matrix 
	//the size will be <num_inputs, num_outputs>
        vector<vector<float>> weights;
	
	//likewise the bias will be vectors, size <num_outputs>
	vector<float> bias;
	
	//the activations, deltas, and errors will be matrices as well
	//size <batch_size, n_outputs>
	vector<vector<float>> activations;
	vector<vector<float>> deltas;
	vector<vector<float>> errors;
	//basically got rid of the neuron struct
	
    public:
        LinearLayer(int, int, float=0.3, int=32);
        void initializeLayer();
        // Main Components
        void forward(vector<vector<float>>);
	void computeDeltas(vector<vector<float>>, vector<vector<float>>);
        void updateWeights(vector<vector<float>>);
        void updateWeightsLegacy(vector<vector<float>>);
	void zeroGrad();
	
        // Helper Classes
        // Printers
	void printActivations();
	void printWeights();
	void printBias();
        //void printNodeWeights(struct node);
        // Setters
	void setLearningRate(float);
        // Getters 
	vector<vector<float>> getActivations(){return this->activations;};
	vector<vector<float>> getDeltas(){return this->deltas;};
	vector<vector<float>> getWeights(){return this->weights;};
	vector<float> getBias();
	float getLearningRate(){return this->learning_rate;};
	int getNumInputs(){return this->num_inputs;};
	int getNumNeurons(){return this->num_outputs;};

	// Setters
	void setActivations(vector<vector<float>> act){this->activations = act;};
	void setDeltas(vector<vector<float>> deltas){this->deltas = deltas;};
	void setErrors(vector<vector<float>> errors){this->errors = errors;};
	
};

#endif

