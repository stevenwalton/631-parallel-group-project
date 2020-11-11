#include <iostream>
#include <vector>
#include "nn.h"
#include "types.h"

int main()
{
    LinearLayer l0(1,2);
    LinearLayer l1(2,1);
    std::vector<struct node> n;
    node nn;
    nn.activation = 1;
    nn.weight.push_back(1);

    n.emplace_back(nn);
    l0.Forward(n, 0);
    //l1.Connect(l0.Get_Output());

    return 0;
}
