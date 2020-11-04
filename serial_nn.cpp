#include <iostream>
#include <list>
#include <cstdlib>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

// Add some graphs to see training progress

double sigmoid(double x){ 
    return 1 / (1 + exp(-x)); 
}

double dSigmoid(double x){ 
    return x * (1 - x); 
}

double dTanh(double x){ 
    return 1 - pow(x, 2); 
}

double unitRandom(){ 
    return ((double)rand())/((double)RAND_MAX); 
}

double crossEntropyLoss(double y, double y_hat){
    return -(y * log(y_hat) + (1-y) * log(1 - y_hat));
}

int main(int argc, const char * argv[]) {
    
    srand(time(0));

    double learningRate;
    int nIterations;
    int interval;
    
    if (argc == 4){
        learningRate = stof(argv[1]);
	nIterations = stoi(argv[2]);
	interval = stoi(argv[3]);
    }else{
	cout << "Usage: \n\t./nn <lr> <n_iter> <interval>\n";
	cout << "Using default values: \n\n";
	cout << "Learning rate = 0.03\n";
	cout << "Iterations = 1000\n";
	cout << "Interval = 1\n\n";
        learningRate = 0.03f;
	nIterations = 1000;
        interval = 1;
    }
    //for (int i = 0; i < argc; ++i) 
    //    cout << argv[i] << "\n"; 
  

    //Reading the dataset
    string line;
    string element;
    ifstream myfile ("circles_dataset.txt");
    double trainingInputs[1000][4];
    double labels[1000];
    int i = 0;
    if (myfile.is_open()){
        while ( getline (myfile, line) ){
            stringstream ss(line);
            getline(ss, element, ',');
            trainingInputs[i][0] = stod(element);
            
	    getline(ss, element, ',');
            trainingInputs[i][1] = stod(element);
            
            //a little bit of feature engineering to help out :p	    
	    trainingInputs[i][2] = pow(trainingInputs[i][0],2);
	    trainingInputs[i][3] = pow(trainingInputs[i][0],2);

	    getline(ss, element, ',');
            labels[i] = stod(element);
            i++;
        }
        myfile.close();
    }

    else cout << "Unable to open file";

    //Network structure
    static const int numInputs = 4;
    static const int numHiddenLayers = 2;
    static const int numHiddenNodes = 2;
    static const int numOutputs = 1;
    
    double hiddenLayer[numHiddenLayers][numHiddenNodes];
    double outputLayer[numOutputs];
    
    double hiddenLayerBias[numHiddenLayers][numHiddenNodes];
    double outputLayerBias[numOutputs];

    double inputHiddenWeights[numInputs][numHiddenNodes];
    double hiddenWeights[numHiddenLayers-1][numHiddenNodes][numHiddenNodes]; //one less cause I separate the input hidden layer
    double outputWeights[numHiddenNodes][numOutputs];
    
    // ********************************** Initializing the weights ********************************************** 
    //Initializing input hidden weights 
    for (int i=0; i<numInputs; i++) {
        for (int j=0; j<numHiddenNodes; j++) {
            inputHiddenWeights[i][j] = unitRandom();
        }
    }

    //Initializing hidden weights
    for (int l = 0; l < numHiddenLayers-1; l++){
        for (int i=0; i<numHiddenNodes; i++) {
            for (int j=0; j<numHiddenNodes; j++) {
                hiddenWeights[l][i][j] = unitRandom();
            }
        }
    }

    //Initializing hidden bias
    for (int l = 0; l < numHiddenLayers; l++){
        for (int z=0; z<numHiddenNodes; z++) {
            hiddenLayerBias[l][z] = unitRandom();
        }
    }

    //Initializing output weights
    for (int i=0; i<numHiddenNodes; i++) {
        for (int j=0; j<numOutputs; j++) {
            outputWeights[i][j] = unitRandom();
        }
    }

    //Initializing output bias
    for (int i=0; i<numOutputs; i++) {
        outputLayerBias[i] = unitRandom();
    }
    

    // ****************************************** Training ***********************************************************
    int r = 0;
    for (int n=0; n < nIterations; n++) {
            
	    //Picking a random example from dataset
	    r = (rand() % 1000); 
            // ******************************************** Forward pass ********************************************************
            
	    //Computing first hidden layer activations from inputs
            for (int j=0; j<numHiddenNodes; j++) {
                double activation = hiddenLayerBias[0][j];
                 for (int k=0; k<numInputs; k++) {
                    activation+=trainingInputs[r][k]*inputHiddenWeights[k][j];
                }
                hiddenLayer[0][j] = sigmoid(activation);
            }
            
	    //Computing intermediate hidden layers
	    //For now, we assume all hidden layers have the same # neurons
	    for (int l = 1; l < numHiddenLayers; l++){
	        for (int j=0; j < numHiddenNodes; j++){
	            double activation = hiddenLayerBias[l][j];
		    for (int k=0; k < numHiddenNodes; k++) {
			// previous layer activation times the corresponding weight
		        activation+=hiddenLayer[l-1][k] * hiddenWeights[l-1][k][j]; //l-1 here cause first hidden layer weights are separate
		    }

		    hiddenLayer[l][j] = sigmoid(activation);
	        }
	    }

	    //Computing output layer from last hidden layer
            for (int j=0; j<numOutputs; j++) {
                double activation=outputLayerBias[j];
                for (int k=0; k<numHiddenNodes; k++) {
                    activation+=hiddenLayer[numHiddenLayers-1][k]*outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
                //outputLayer[j] = activation; //doing regression
            }
            
           // *************************************************************** Printing results ********************************************************************

	    if(n % interval == 0){ 
                //cout << "Iteration "<< n <<" Input:" << trainingInputs[r][0] << " " << trainingInputs[r][1] << "    Output:" << outputLayer[0] << "    Expected Output: " << labels[r] << "\n";
                //cout << "x = " << trainingInputs[r][0] << "y = " << trainingInputs[r][1] << "\n";
		cout << "Iteration " << n << " Output:" << outputLayer[0] << " Expected Output: " << labels[r] << " Loss: " <<  crossEntropyLoss(labels[r], outputLayer[0]) <<"\n"; 
	    }
	    
           // *************************************************************** Backpropagation **********************************************************************
            
	    //Finding the updates for the output layer
            double deltaOutput[numOutputs];
            for (int j=0; j<numOutputs; j++) {
                double errorOutput = labels[r] - outputLayer[j];
                deltaOutput[j] = errorOutput * dSigmoid(outputLayer[j]);
         	//cout << "delta output = " << deltaOutput[j] << "\n";
            }
            
	    //Finding the updates for the last hidden layer
            double deltaHidden[numHiddenLayers][numHiddenNodes];
	    for (int j=0; j < numHiddenNodes; j++) {
                double errorHidden = 0.0f;
                for(int k=0; k < numOutputs; k++) {
                    errorHidden += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[numHiddenLayers-1][j] = errorHidden * dSigmoid(hiddenLayer[numHiddenLayers-1][j]);
            }
            
            //Finding the updates for the rest of hidden layers
	    for (int l = numHiddenLayers - 2; l >= 0; l--){
	        for (int j=0; j < numHiddenNodes; j++) {
                    double errorHidden = 0.0f;
                    for(int k=0; k < numHiddenNodes; k++) {
                        errorHidden += deltaHidden[l+1][k] * hiddenWeights[l][j][k];
                    }
                    deltaHidden[l][j] = errorHidden * dSigmoid(hiddenLayer[l][j]);
                }
	    }

            //output bias and weight updates
            for (int j=0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * learningRate;
                for (int k=0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[numHiddenLayers-1][k] * deltaOutput[j] * learningRate;
                }
            }
            
            //rest of hidden layers bias and weight updates
            for (int l = numHiddenLayers-1; l > 0; l--){
                for (int j=0; j < numHiddenNodes; j++) {
                    hiddenLayerBias[l][j] += deltaHidden[l][j] * learningRate;
                    for(int k=0; k < numHiddenNodes; k++) {
                        hiddenWeights[l-1][k][j] += hiddenLayer[l-1][k] * deltaHidden[l][j] * learningRate; //here hiddenWeights[l-1] because input hidden weights are separate
                    }
                }
	    }

            //input hidden layer bias and weight updates
            for (int j=0; j < numHiddenNodes; j++) {
                hiddenLayerBias[0][j] += deltaHidden[0][j] * learningRate;
                for(int k=0; k < numInputs; k++) {
                    inputHiddenWeights[k][j] += trainingInputs[r][k] * deltaHidden[0][j] * learningRate;
                }
            }
   
    }
 
    return 0;
}
