#include "utils.h"
#include<algorithm>
#include<armadillo>

using namespace std;
using namespace arma;

void MatMul(double* const mat1, double* const mat2, double *result, int nrow1, int ncol1, int ncol2)
{
    const mat m1(mat1,nrow1,ncol1,false);
    mat m2(mat2,ncol1,ncol2,false);
    mat res(result,nrow1,ncol2,false);
    res=m1*m2;
}

void MatFullDiag(double* const X,int n,double *evec,double *eval)
{
    const mat mX(X,n,n,false);
    mat mevec(evec,n,n,false);
    vec meval(eval,n,false);
    eig_sym(meval,mevec,mX);
}
