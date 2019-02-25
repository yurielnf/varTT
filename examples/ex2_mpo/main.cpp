#include <iostream>
#include"mps.h"
#include<cassert>

using namespace std;

int main()
{
    cout << "Hello World!" << endl;

    MPS x(10,20);
    x.FillRandu({20,2,20});


    return 0;
}
