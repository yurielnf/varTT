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


array<stdvec,2> MatSVD(bool is_right,const double*  X, int n1,int n2,double tol,int Dmax)
{
    const mat mX((double* const)X,n1,n2,false);
    mat U,V;
    vec s;
    svd_econ(U, s, V, mX);
    int D=std::min(Dmax,(int)s.size());
    if (D==0)
    {
        D=s.size();}
        for(uint i=0;i<D;i++)
            if(fabs(s[i])<=tol) {D=i;break;}

    double sum=0;
    for(int i=0;i<D;i++) sum+=s[i]*s[i];
    mat Ua=U.head_cols(D);
    mat Sa=diagmat(s.head(D)/sum);
    mat Vt=V.head_cols(D).t();
    if (is_right)
        return {conv_to<stdvec>::from(vectorise(Ua*Sa)),
                conv_to<stdvec>::from(vectorise(Vt))    };
    else
        return {conv_to<stdvec>::from(vectorise(Ua)),
                conv_to<stdvec>::from(vectorise(Sa*Vt)) };
}

static bool abs_compare(double a, double b)
{
    return (std::abs(a) < std::abs(b));
}

vector<int> FindNonZeroCols(const double*  X, int n1,int n2,double tol)
{
    vector<int> cols;
    for(int j=0;j<n2;j++)
    {
        double mc=*std::max_element(X+j*n1,X+(j+1)*n1,abs_compare);
        if ( std::abs(mc) > tol ) cols.push_back(j);
    }
    if (cols.empty())
        throw runtime_error("FindNonZeroCols with cero col");
    return cols;
}

mat MatSelectCols(const mat& A, const vector<int>& cols)
{
    mat B(A.n_rows,cols.size());
    for(uint j=0;j<cols.size();j++)
        B.col(j)=A.col(cols[j]);
    return B;
}

array<mat,2> MatChopDecompArmaByCol(const mat& mX,double tol)
{
    int n1=mX.n_rows, n2=mX.n_cols;
    mat U(n2,n2,fill::eye);
    auto cols=FindNonZeroCols(mX.memptr(),n1,n2,tol);
    mat Xa=MatSelectCols(mX,cols);
    mat Ua=MatSelectCols(U ,cols).t();
    return {Xa,Ua};
}

array<stdvec,2> MatChopDecomp(bool is_right, const double*  X, int n1, int n2, double tol)
{
    const mat mX((double* const)X,n1,n2,false);
//    tol=max( tol, mX.max()*numeric_limits<double>::epsilon());
    if (is_right)
    {
//        stdvec vx(X,X+n1*n2);
//        stdvec vu(n1*n1);
//        MatFillEye(vu.data(),n1);
//        return {vu,vx};

        auto ab=MatChopDecompArmaByCol(mX.t(),tol);
        mat a=ab[1].t();
        mat b=ab[0].t();
        return {stdvec(a.begin(),a.end()),
                stdvec(b.begin(),b.end())};
    }
    else
    {
//        stdvec vx(X,X+n1*n2);
//        stdvec vu(n2*n2);
//        MatFillEye(vu.data(),n2);
//        return {vx,vu};

        auto ab=MatChopDecompArmaByCol(mX,tol);
        return {stdvec(ab[0].begin(),ab[0].end()),
                stdvec(ab[1].begin(),ab[1].end())};
    }
}

array<stdvec,2> MatQRDecomp(bool is_right, const double*  X, int n1, int n2)
{
    const mat mX((double* const)X,n1,n2,false);
    if (is_right)
    {
        mat Q,R;
        qr_econ(Q,R,mX.t());
        mat a=R.t();
        mat b=Q.t();
        return {stdvec(a.begin(),a.end()),
                stdvec(b.begin(),b.end())};
    }
    else
    {
        mat Q,R;
        qr_econ(Q,R,mX);
        return {stdvec(Q.begin(),Q.end()),
                stdvec(R.begin(),R.end())};
    }
}


//void CubeTranspose(const double*  X, double *result, int d1, int d2,int d3)
//{
//    const cube mX((double* const)X,d1,d2,d3,false);
//    cube res(result,d3,d2,d1,false);
//    res=mX.t();
//}
