#ifndef UTILS_H
#define UTILS_H

#include<vector>
#include<array>
#include<random>
#include<complex>
#include<iomanip>
#include<iostream>

typedef std::vector<double> stdvec;

template<class T>
T VecReduce(T* v,int n)
{
    int step=1;
    while (n>step)
    {
        for(int i=0;i<n;i+=2*step)
            if (i+step<n)
                v[i]+=v[i+step];
        step+=step;
    }
    return v[0];
}

template<class T>
void VecFillRandu(T *vec, int n)
{
    for(int i=0;i<n;i++)
        vec[i]=T(1.0*rand()/RAND_MAX);
}
template<class T>
void VecFillZeros(T *vec, int nelem)
{
    for(int i=0;i<nelem;i++)
        vec[i]=T(0);
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
double VecNorm( T* const vec, int nelem)
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
T VecDot(const T* vec1,const T* vec2,int n)
{
    T s=T(0);
    for(int i=0;i<n;i++)
        s+=std::conj(vec1[i]) * vec2[i];
    return s;
}

//---------------------------------- Matrix -----------------------
template<class T>
void MatFillEye(T *dat, int n)
{
    VecFillZeros(dat,n*n);
    for(int i=0;i<n;i++)
        dat[i+i*n]=1;
}
void MatTranspose(const double*  X, double *result, int nrow, int ncol);
void MatMul(const double*  mat1,const double*  mat2, double *result, int nrow1, int ncol1, int ncol2);
void MatFullDiag(double * const X, int n, double *evec, double *eval);
std::array<stdvec,2> MatSVD(bool is_right,const double*  X, int n1,int n2,double tol);
std::array<stdvec,2> MatChopDecomp(bool is_right,const double*  X, int n1,int n2,double tol);


//--------------------------------- Cube --------------------------

//void CubeTranspose(const double*  X, double *result, int d1, int d2,int d3);



#endif // UTILS_H
