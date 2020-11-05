#include <iostream>
#include <list>
#include <cstdlib>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <bits/stdc++.h>

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
    int nEpochs;
    double trainSplit;
    int interval = 200;
    
    if (argc == 4){
        learningRate = stof(argv[1]);
	nEpochs = stoi(argv[2]);
	trainSplit = stof(argv[3]);
    }else{
	cout << "Usage: \n\t./nn <lr> <n_epochs> <train_split>\n";
	cout << "Using default values: \n\n";
	cout << "Learning rate = 0.1\n";
	cout << "Epochs = 1000\n";
	cout << "Train split = 0.8\n";
        learningRate = 0.1f;
	nEpochs = 1000;
	trainSplit = 0.8;
    }
  

    //Reading the dataset
    string line;
    string element;
    ifstream myfile ("circles_dataset.txt");
    int dataset_size = 1000;
    double trainingInputs[dataset_size][2];
    double labels[dataset_size][1];
    int trainSize = int(trainSplit * dataset_size);
    
    for (int i = 0; i < dataset_size; i++) {
        for (int j=0; j<1; j++) {
            labels[i][j] = 0.0;
        }
    }

    int i = 0;
    if (myfile.is_open()){
        while ( getline (myfile, line) ){
            stringstream ss(line);
            getline(ss, element, ',');
            trainingInputs[i][0] = stod(element);
            
	    getline(ss, element, ',');
            trainingInputs[i][1] = stod(element);
            
            //a little bit of feature engineering to help out :p	    
	    //trainingInputs[i][2] = pow(trainingInputs[i][0],2);
	    //trainingInputs[i][3] = pow(trainingInputs[i][0],2);

	    getline(ss, element, ',');
            //labels[i][stoi(element)] = 1.0;
	    labels[i][0] = stod(element);
            i++;
        }
        myfile.close();
    }
    else cout << "Unable to open file";

    vector<int> arr;

    // creating a an array to shuffle for training
    for (int j = 0; j < dataset_size; ++j)
        // 1 2 3 4 5 6 7 8 9
        arr.push_back(j);

    // using built-in random generator
    random_shuffle(arr.begin(), arr.end());


    //Network structure
    static const int numInputs = 2;
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
    int index;
    double epochLoss;
    int epochHits;
    for (int n=0; n < nEpochs; n++){
        
	epochLoss = 0.0;
	epochHits = 0;
        
	for (r = 0; r < trainSize; r++){
	
            index = arr[r];
	    //Picking a random example from dataset
	    //r = (rand() % 1000); 
            // ******************************************** Forward pass ********************************************************
            
	    //Computing first hidden layer activations from inputs
            for (int j = 0; j < numHiddenNodes; j++) {
                double activation = hiddenLayerBias[0][j];
                 for (int k = 0; k < numInputs; k++) {
                    activation += trainingInputs[index][k] * inputHiddenWeights[k][j];
                }
                hiddenLayer[0][j] = sigmoid(activation);
            }
            
	    //Computing intermediate hidden layers
	    //For now, we assume all hidden layers have the same # neurons
	    for (int l = 1; l < numHiddenLayers; l++){
	        for (int j = 0; j < numHiddenNodes; j++){
	            double activation = hiddenLayerBias[l][j];
		    for (int k = 0; k < numHiddenNodes; k++) {
			// previous layer activation times the corresponding weight
		        activation += hiddenLayer[l-1][k] * hiddenWeights[l-1][k][j]; //l-1 here cause first hidden layer weights are separate
		    }

		    hiddenLayer[l][j] = sigmoid(activation);
	        }
	    }

	    //Computing output layer from last hidden layer
            for (int j=0; j < numOutputs; j++) {
                double activation = outputLayerBias[j];
                for (int k = 0; k < numHiddenNodes; k++) {
                    activation += hiddenLayer[numHiddenLayers-1][k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
                //outputLayer[j] = activation; //doing regression
            }
            
           // *************************************************************** Printing results ********************************************************************
           /*
	    if(r % interval == 0){ 
                //cout << "Iteration "<< n <<" Input:" << trainingInputs[index][0] << " " << trainingInputs[index][1] << "    Output:" << outputLayer[0] << "    Expected Output: " << labels[index] << "\n";
                //cout << "x = " << trainingInputs[index][0] << "y = " << trainingInputs[index][1] << "\n";
		//cout << "Iteration " << n << " : " << r << " Output:" << ((outputLayer[0] > outputLayer[1]) ? 0 : 1) << " Expected Output: " << ((labels[index][0] > labels[index][1]) ? 0 : 1) << "\n";//" Loss: " <<  crossEntropyLoss(labels[index], outputLayer[0]) <<"\n"; 
	        //cout << "Iteration " << n << " : " << r << " Output:" << (outputLayer[0]) << " Expected Output: " << (labels[index][0]) << " Loss: " <<  crossEntropyLoss(labels[index][0], outputLayer[0]) <<"\n";
	    }
	    */
	    epochLoss += crossEntropyLoss(labels[index][0], outputLayer[0]);
	    if (round(outputLayer[0]) == labels[index][0]) epochHits += 1;
	    
           // *************************************************************** Backpropagation **********************************************************************
            
	    //Finding the updates for the output layer
            double deltaOutput[numOutputs];
	    double errorOutput;
            for (int j=0; j<numOutputs; j++) {
                errorOutput = labels[index][j] - outputLayer[j];
                deltaOutput[j] = errorOutput * dSigmoid(outputLayer[j]);
         	//cout << "delta output = " << deltaOutput[j] << "\n";
            }
            
	    //Finding the updates for the last hidden layer
            double deltaHidden[numHiddenLayers][numHiddenNodes];
	    double errorHidden;
	    for (int j=0; j < numHiddenNodes; j++) {
                errorHidden = 0.0f;
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
                    inputHiddenWeights[k][j] += trainingInputs[index][k] * deltaHidden[0][j] * learningRate;
                }
            }
        }
        cout << "Epoch " << n << " Train Accuracy: " << (double)epochHits/(double)trainSize << " Loss: " << epochLoss/trainSize << endl;	
    }
 
    return 0;
}
