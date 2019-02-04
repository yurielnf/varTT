#ifndef UTILS_H
#define UTILS_H

#include<vector>
#include<random>
#include<complex>
#include<iomanip>
#include<iostream>

template<class T>
void VecFillRandu(T *vec, int nelem)
{
    for(int i=0;i<nelem;i++)
        vec[i]=T(1.0*rand()/RAND_MAX);
}

template<class T>
void VecSave(const T* vec, int nelem, std::ostream& out)
{
    out<<std::setprecision(17);
    for(int i=0;i<nelem;i++)
        out<<vec[i]<<" ";
    out<<"\n";
}

template<class T>
void VecLoad(T* vec, int nelem, std::istream& in)
{
    for(int i=0;i<nelem;i++)
        in>>vec[i];
}

template<class T>
double VecNorm(const T* vec, int nelem)
{
    double sum=0;
    for(int i=0;i<nelem;i++)
        sum+=std::norm(vec[i]);
    return sqrt(sum);
}

template<class T>
void VecProd(T* vec, int n, T c)
{
    for(int i=0;i<n;i++)
        vec[i]*=c;
}

template<class T>
void VecPlusInplace(T* vec,const T* vec2,int n)
{
    for(int i=0;i<n;i++)
        vec[i]+=vec2[i];
}

template<class T>
void VecMinusInplace(T* vec,const T* vec2,int n)
{
    for(int i=0;i<n;i++)
        vec[i]-=vec2[i];
}

template<class T>
void VecNegativeInplace(T* vec,int n)
{
    for(int i=0;i<n;i++)
        vec[i]=-vec[i];
}

template<class T>
T VecDot(T* vec1,const T* vec2,int n)
{
    T s=T(0);
    for(int i=0;i<n;i++)
        s+=std::conj(vec1[i]) * vec2[i];
    return s;
}

//---------------------------------- Matrix -----------------------

void MatMul(const double*  mat1,const double*  mat2, double *result, int nrow1, int ncol1, int ncol2);
void MatFullDiag(double * const X, int n, double *evec, double *eval);
void MatSVD(const double*  X, int m,int n,double *U,double *S,double *V);


#endif // UTILS_H
