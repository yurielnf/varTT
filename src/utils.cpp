#include "utils.h"
#include<armadillo>

using namespace std;
using namespace arma;

void MatTranspose(const double*  X, double *result, int nrow, int ncol)
{
    const mat mX((double* const)X,nrow,ncol,false);
    mat res(result,ncol,nrow,false,true);
    res=mX.t();
}

void MatMul(const double*  mat1, const double*  mat2, double *result, int nrow1, int ncol1, int ncol2)
{
    const mat m1((double* const)mat1,nrow1,ncol1,false);
    const mat m2((double* const)mat2,ncol1,ncol2,false);
    mat res(result,nrow1,ncol2,false);
    res=m1*m2;
}

void MatFullDiag(double* const X,int n,double *evec,double *eval)
{
    const mat mX(X,n,n,false);
    mat mevec(evec,n,n,false,true);
    vec meval(eval,n,false);
    eig_sym(meval,mevec,mX);
}

//void MatSVD(const double*  X, int m,int n,double *U,double *S,double *Vt)
//{
//    const mat mX((double* const)X,m,n,false);
//    int ns=std::min(m,n);
//    mat mU(U,m,ns,false);
//    mat vS(S,ns,ns,false);
//    mat mVt(Vt,ns,n,false),V;
//    vec s;
//    svd_econ(mU, s, V, mX);
//    vS.diag()=s;
//    mVt=V.t();
//}

//void MatSVD(const double*  X, int m,int n,double *U,double *S,double *V)
//{
//    const mat mX((double* const)X,m,n,false);
//    int ns=std::min(m,n);
//    mat mU(U,m,ns,false,true);
//    vec vS(S,ns,false,true);
//    mat mV(V,n,ns,false,true);
//    svd_econ(mU, vS, mV, mX);
//}

vector<vector<double>> MatSVD(const double*  X, int n1,int n2,double tol)
{
    const mat mX((double* const)X,n1,n2,false);
    mat U,V;
    vec s;
    svd_econ(U, s, V, mX);
    tol=max( tol, s[0]*numeric_limits<double>::epsilon());
    int D=s.size();
    for(int i=0;i<D;i++)
        if(i>0 && fabs(s[i])<tol) {D=i;break;}

    vector<double> A(n1*D), c(D), B(n2*D);
    mat Aa(A.data(),n1,D,false,true);
    vec ca(c.data(),D,false,true);
    mat Ba(B.data(),n2,D,false,true);
    Aa=U.head_cols(D);
    ca=s.head(D);
    Ba=V.head_cols(D);
    return {A,c,B};
}


//void CubeTranspose(const double*  X, double *result, int d1, int d2,int d3)
//{
//    const cube mX((double* const)X,d1,d2,d3,false);
//    cube res(result,d3,d2,d1,false);
//    res=mX.t();
//}
