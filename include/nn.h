#ifndef NN_H
#define NN_H

#include <array>
#include <vector>
#include "types.h"

class LinearLayer
{
    private:
        int num_input;
        int num_output;

        std::vector<struct node> input_nodes;
        std::vector<struct node> output_nodes;

    public:
        LinearLayer(int, int);
        void InitializeLayer();
        void Forward();
        void Backward();
        void Connect(std::vector<struct node>);
        std::vector<struct node> Get_Output(){return output_nodes;}

};

#endif
