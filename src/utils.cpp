#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>

void readDataset(std::string filename, std::vector<std::vector<float>> &features, std::vector<int> &labels)

{
    //Reading the dataset
	std::string line;
	std::string element;
	std::ifstream myfile (filename);

    float x,y;

    if (myfile.is_open()){
        while ( getline (myfile, line) ){

		std::stringstream ss(line);
            	getline(ss, element, ',');
            	x = stof(element);
            	getline(ss, element, ',');	
		y = stof(element);

		std::vector<float> feature_instance{x, y};
	    	features.emplace_back(feature_instance);

            	getline(ss, element, ',');
	    	//std::vector<float> label_instance{stof(element)};
	    	labels.emplace_back(stoi(element));
	}
        myfile.close();
    }
    else{
        std::cout << "Unable to open file";
    }
    std::cout << "Dataset loaded: " << features.size() << " rows"<< std::endl;
}

void printFloatVector(std::vector<float> v)
{
	for(float f : v)
		std::cout << f << " ";
	std::cout << "\n";
}

void printFloatMatrix(std::vector<std::vector<float>> m)
{
	for(std::vector<float> v : m)
		printFloatVector(v);
}
