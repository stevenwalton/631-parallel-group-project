#include <iostream>
#include <vector>
#include <cassert>
#include <time.h>
#include <string>
#include <cmath>
#include <algorithm>
#include "nn.h"
#include "sequential.h"
#include "utils.h"

#include "cuda_math.h"

using namespace std;

double crossEntropyLoss(double y, double y_hat){
    return -(y * log(y_hat) + (1-y) * log(1 - y_hat));
}

void defineModel(Sequential &model, float learning_rate)
{
    //creating the layers
    LinearLayer inputLayer(2,3, learning_rate, model.getBatchSize());
    LinearLayer h1(3,3, learning_rate, model.getBatchSize());
    LinearLayer h2(3,3, learning_rate, model.getBatchSize());
    LinearLayer outputLayer(3, 1, learning_rate, model.getBatchSize());

    //Adding the layers to the model
    model.add(inputLayer);
    model.add(h1);
    //model.add(h2);
    model.add(outputLayer);
}

int main(int argc, const char * argv[])
{
    //setting a random seed
    srand (time(0));
    //srand(12345);
    string dataset;
    int n_epochs;
    float learning_rate;
    int batch_size;

    if (argc == 5){
        dataset = argv[1];
        learning_rate = stof(argv[2]);
        n_epochs = stoi(argv[3]);
        batch_size = stoi(argv[4]);
    }else{
        cout << "Usage: \n\t./nn <dataset_file> <lr> <n_epochs>\n\n";
        cout << "Using default values: \n\n";
        cout << "Dataset = moons_dataset.txt\n";
        cout << "Learning rate = 0.3\n";
        cout << "Epochs = 500\n";
        cout << "Batch size = 8\n\n";
        dataset = "datasets/moons_dataset.txt";
    	learning_rate = 0.5f;
        n_epochs = 500;
        batch_size = 8;
    }

    //vectors to hold the data
    vector<vector<float>> features;
    vector<int> labels;

    readDataset(dataset, features, labels);

    /************* Define Model *********************/
    Sequential model;
    model.setBatchSize(batch_size);
    defineModel(model, learning_rate);
    cout << endl;
    model.printModel();

    //float epochLoss;
    int epochHits; 
    vector<float> predictions;
    vector<vector<float>> batch_preds;
    int batch_loops = features.size() / batch_size;
    
    /************* Training  *********************/

    cout << "Starting training, using learning rate = " << model.getLayers()[0].getLearningRate() << "\n";
    
    float accuracy;
    int prediction;
    for (int n = 0; n < n_epochs; n++)
    {
	//cout << "Epoch " << n << endl;
        epochHits = 0;
        for (int i = 0; i < batch_loops; ++i)
        {
	    //cout << "Batch " << i+1 << "/" << batch_loops << endl;
            vector<vector<float>> feature_vec(batch_size);
            vector<int> label_vec(batch_size);
            for (int j = 0; j < batch_size; ++j)
            {
                feature_vec[j] = features[(i*batch_size)+j];
                label_vec[j] = labels[(i*batch_size)+j];
            }
            model.trainIteration(feature_vec, label_vec);
            batch_preds = model.forward(feature_vec);
            for (int j = 0; j < batch_size; ++j)
            {
                if(round(batch_preds[j][0]) == label_vec[j])
                    epochHits += 1;

		//prediction = std::distance(batch_preds[j].begin(), std::max_element(batch_preds[j].begin(), batch_preds[j].end()));
		//if( prediction == label_vec[j])
		//	epochHits += 1;
            }
        }

        /*
        for (size_t i = 0; i < features.size(); ++i)
        {
            predictions = model.forward(features[i]);
            if (round(predictions[0]) == labels[i][0])
                epochHits += 1;
        }
        */

        accuracy = (float)epochHits/(float)features.size();
        cout << "Epoch " << n << " Accuracy: " << accuracy << "\n";
    	if(accuracy == 1)
        {
        	cout << "Reached accuracy of 1. Stopping early" << endl;
        	break;
        }
    }
    return 0;
}
