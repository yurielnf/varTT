#include "utils.h"
#include<algorithm>
#include<armadillo>

using namespace std;
using namespace arma;

int Prod(std::vector<int> dim, int pos)
{
    return accumulate(dim.begin(),dim.begin()+pos,1,multiplies<int>());
}

int Offset(std::vector<int> id, std::vector<int> dim)
{
    int sum=0,prod=1;
    for(uint i=0;i<id.size();i++)
    {
        sum+=id[i]*prod;
        prod*=dim[i];
    }
    return sum;
}


void MatMul(double *mat1, double *mat2, double *result, int nrow1, int ncol1, int ncol2)
{
    mat m1(mat1,nrow1,ncol1,false);
    mat m2(mat2,ncol1,ncol2,false);
    mat res(result,nrow1,ncol2,false);
    res=m1*m2;
}
