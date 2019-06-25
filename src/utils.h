#ifndef UTILS_H
#define UTILS_H

#include<vector>
#include<array>
#include<random>
#include<complex>
#include<iomanip>
#include<iostream>

typedef std::vector<double> stdvec;
typedef std::complex<double> cmpx;

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
//    if (binary)
//        out.write(reinterpret_cast<const char*>(vec),nelem*sizeof(T));
//    else
//    {
        out<<std::setprecision(17);
        for(int i=0;i<nelem;i++)
            out<<vec[i]<<" ";
//    }
    out<<"\n";
}

template<class T>
void VecLoad(T* vec, int nelem, std::istream& in)
{
//    if (binary)
//        in.read(reinterpret_cast<char*>(vec),nelem*sizeof(T));
//    else
//    {
        for(int i=0;i<nelem;i++)
            in>>vec[i];
//    }
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
        s+=std::real(std::conj(vec1[i]) * vec2[i]);
    return s;
}

template<class T>
std::vector<T> UniformPartition(const T& x1,const T& x2, int nX)
{
    std::vector<T> res(nX);
    T dx=(x2-x1)/T(nX-1);
    for(int i=0;i<nX;i++) res[i]=x1+dx*T(i);
    return res;
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
std::array<stdvec,2> MatSVD(bool is_right, const double*  X, int n1, int n2, double tol, int Dmax);
std::array<stdvec,2> MatChopDecomp(bool is_right, const double*  X, int n1, int n2, double tol);
std::array<stdvec,2> MatQRDecomp(bool is_right, const double*  X, int n1, int n2);

struct MatChopDecompFixedTol
{
    double tol;
    MatChopDecompFixedTol(double tol): tol(tol) {}
    std::array<stdvec,2> operator()(bool is_right,const double*  X, int n1,int n2) const
    {
        return MatChopDecomp(is_right,X,n1,n2,tol);
    }
};
struct MatChopDecompFixedDim // <--------------------------- to be continue
{
    int d;
    MatChopDecompFixedDim(int d):d(d){}
    std::array<stdvec,2> operator()(bool is_right,const double*  X, int n1,int n2) const
    {
        return MatChopDecomp(is_right,X,n1,n2,0);
    }
};
struct MatSVDFixedTol
{
    double tol;
    MatSVDFixedTol(double tol):tol(tol){}
    std::array<stdvec,2> operator()(bool is_right,const double*  X, int n1,int n2) const
    {
        return MatSVD(is_right,X,n1,n2,tol,0);
    }
};
struct MatSVDFixedDim
{
    int d;
    MatSVDFixedDim(int d):d(d){}
    std::array<stdvec,2> operator()(bool is_right, const double*  X, int n1,int n2) const
    {
        return MatSVD(is_right,X,n1,n2,-1,d);
    }
};

struct MatSVDAdaptative
{
    double tol;
    int d;
    MatSVDAdaptative(double tol,int d): tol(tol), d(d) {}
    std::array<stdvec,2> operator()(bool is_right, const double*  X, int n1,int n2) const
    {
        return MatSVD(is_right,X,n1,n2,tol,d);
    }
};

struct MatSVDFixedDimSE
{
    int d;
    stdvec P;
    MatSVDFixedDimSE(int d,stdvec P):d(d),P(P){}
    std::array<stdvec,2> operator()(bool is_right, const double*  X, int n1,int n2) const
    {
        std::array<stdvec,2> AB;
        stdvec Xb(n1*n2+P.size());
        if (is_right)
        {
            int n1xb=Xb.size()/n2;
            int n1p=P.size()/n2;
            for(int i=0;i<n1;i++)
                for(int j=0;j<n2;j++)
                    Xb[i+j*n1xb]=X[i+j*n1];    //col-major
            for(int i=0;i<n1p;i++)
                for(int j=0;j<n2;j++)
                    Xb[i+n1+j*n1xb]=P[i+j*n1p];    //col-major

            AB=MatSVD(is_right,Xb.data(),n1xb,n2,-1,d);
            int m=AB[0].size()/n1xb;
            stdvec a(n1*m);
            for(int i=0;i<n1;i++)
                for(int j=0;j<m;j++)
                    a[i+j*n1]=AB[0][i+j*n1xb];    //col-major
            AB[0]=a;
        }
        else
        {
            std::copy(X,X+n1*n2,Xb.begin());
            std::copy(P.begin(),P.end(),Xb.begin()+n1*n2);
            int n2p=P.size()/n1;
            int n2xb=n2+n2p;
            AB=MatSVD(is_right,Xb.data(),n1,n2xb,-1,d);
            AB[1].resize(AB[1].size()*n2/n2xb);
        }
        return AB;
    }
};

struct MatDensityFixedDimDecomp
{
    stdvec rot;
    int m;
    MatDensityFixedDimDecomp(stdvec rho,int m);
    std::array<stdvec,2> operator()(bool is_right, const double*  X, int n1,int n2) const;
};

//--------------------------------- Cube --------------------------

//void CubeTranspose(const double*  X, double *result, int d1, int d2,int d3);



#endif // UTILS_H
