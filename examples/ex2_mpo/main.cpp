#include <iostream>
#include"mps.h"
#include<cassert>

using namespace std;

int main()
{
    cout << "Hello World!" << endl;

    MPS x(10,20);
    x.FillRandu({20,2,20});
        for(int i=0;i<2;i++)
        {
            x.PrintSizes();
            int m=1;
            for(auto t:x.M)
            {
                assert( t.dim[0]==m );
                assert( t.dim[1]==2 );
                assert( t.dim[2]<=x.m );
                m=t.dim[2];
            }
            x.Canonicalize();
        }

    return 0;
}
