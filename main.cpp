#include <iostream>
#include "nn.h"
#include "sequential.h"
#include <vector>
#include <cassert>
#include <time.h>
#include "utils.h"
#include <string>

using namespace std;

double crossEntropyLoss(double y, double y_hat){
    return -(y * log(y_hat) + (1-y) * log(1 - y_hat));
}

void defineModel(Sequential &model, float learning_rate)
{
    //creating the layers
    LinearLayer inputLayer(2,3, learning_rate);
    LinearLayer h1(3,3, learning_rate);
    LinearLayer h2(3,3, learning_rate);
    LinearLayer outputLayer(3,1, learning_rate);

    //Adding the layers to the model
    model.add(inputLayer);
    model.add(h1);
    model.add(h2);
    model.add(outputLayer);
}

int main(int argc, const char * argv[])
{
    //setting a random seed
    srand (time(0));

    string dataset;
    int n_epochs;
    float learning_rate;

    if (argc == 4){
	dataset = argv[1];
        learning_rate = stof(argv[2]);
        n_epochs = stoi(argv[3]);
    }else{
        cout << "Usage: \n\t./nn <dataset_file> <lr> <n_epochs>\n\n";
        cout << "Using default values: \n\n";
	cout << "Dataset = moons_dataset.txt\n";
        cout << "Learning rate = 0.3\n";
        cout << "Epochs = 200\n\n";
        dataset = "moons_dataset.txt";
    	learning_rate = 0.3f;
        n_epochs = 200;
    }

    //vectors to hold the data
    vector<vector<float>> features;
    vector<vector<float>> labels;

    readDataset(dataset, features, labels);

    /************* Define Model *********************/
    Sequential model;
    defineModel(model, learning_rate);
    cout << endl;
    model.printModel();

    float epochLoss;
    int epochHits; 
    vector<float> predictions;
    
    /************* Training  *********************/

    cout << "Starting training, using learning rate = " << model.getLayers()[0].getLearningRate() << "\n";
    
    float accuracy; 
    for (int n = 0; n < n_epochs; n++)
    {
    	epochLoss = 0.0;
	epochHits = 0;
	for (size_t i = 0; i < features.size(); i++)
	{       
		model.trainIteration(features[i], labels[i]);
		predictions = model.forward(features[i]);
		
		//looking for hits
		//this needed to be adapted for multiple outputs
		epochLoss += crossEntropyLoss(labels[i][0], predictions[0]);
		if(round(predictions[0]) == labels[i][0])
			epochHits += 1;
	}
        accuracy = (float)epochHits/(float)features.size();
	cout << "Epoch " << n << " Accuracy: " << accuracy << " Loss: " << epochLoss/features.size()<< "\n";
        if (accuracy == 1)
        {
            cout << "Reached accuracy of 1. Stopping early" << endl;
            break;
        }
    }
    return 0;
}
