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

    if (myfile.is_open()){
        while ( getline (myfile, line) )
        {
            std::stringstream ss(line);
	    std::vector<float> feature_instance;
	    while (ss.good())
	    {
		getline(ss, element, ',');
                feature_instance.emplace_back(stof(element));
	    }
	    
            labels.emplace_back((int)feature_instance[feature_instance.size()-1]);
	    feature_instance.pop_back();
	    features.emplace_back(feature_instance);
	}
        myfile.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }
    //std::cout << "Dataset loaded: " << features.size() << " rows"<< std::endl;
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
