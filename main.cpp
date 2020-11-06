#include "nn.h"

int main()
{
    LinearLayer l0(2,3);
    LinearLayer l1(3,3);
    l0.Forward();
    l1.Connect(l0.Get_Output());

    return 0;
}
