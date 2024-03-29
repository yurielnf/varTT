#ifndef LANCZOS_H
#define LANCZOS_H

#include<iostream>
#include<vector>
#include<array>
#include<stdexcept>

#ifndef MKL
 #include<lapacke.h>
#else
 #include<mkl_lapacke.h>
#endif

using std::vector;
using std::array;

struct LEigenPair
{
    double eval;
    vector<double> evec;
};


template<class Ket>
void Orthogonalize(Ket &t,const std::vector<Ket>& basis,int size)
{
    const double k=0.25, tol=1e-14;
    double tauin=Norm(t);
    for (int i = 0; i < size; ++i)
    {
        auto d=-Dot(basis[i],t);
        if (fabs(d)>tol)
//        t+=basis[i]*(-Dot(basis[i],t));
            t.pexa(basis[i],d);
    }
    if ( Norm(t)/tauin > k ) return;
    for (int i = 0; i < size; ++i)
    {
        auto d=-Dot(basis[i],t);
        if (fabs(d)>tol)
//        t+=basis[i]*(-Dot(basis[i],t));
            t.pexa(basis[i],d);
    }
}

LEigenPair GSTridiagonal(double *a, double *b, int size, double tol=0);
array<stdvec,2> EigenFullTridiagonal(double *a,double *b,int n,double tol=0);
vector<vector<cmpx>> CorrectionVTrid(const vector<cmpx>& ws,double *a,double *b, int n);


template<class LinearOperator, class Ket>
struct Lanczos
        //Find the lowest eigenvalue lambda0, and eigenvector .GetState()
        //for the eigen-problem A x = lambda x, A Hermitian
{
    const LinearOperator& A;
    Ket r;
    vector<Ket> v;      // Orthonormal basis for the Krylov space
    vector<double> a,b; // the tridiagonal matrix: principal and second diagonals
    int iter;

    double lambda0=1, error=1;
    vector<double> evec;

    Lanczos(const LinearOperator& A, const Ket& r0)
        :A(A)
        ,r(r0)
        ,b({Norm(r0)})
        ,iter(0)
    {}

    void Iterate()
    {
        if (b[iter]< fabs(lambda0*std::numeric_limits<double>::epsilon()) )
        {
            r.FillRandu(); r*=(1.0/Norm(r));
            Orthogonalize(r,v,iter-1);
            b[iter]=Norm(r);
        }
        v.push_back( r*(1.0/b[iter]) );
        r=A*v[iter];
        if (iter>0) r.pexa(v[iter-1],-b[iter]);
        a.push_back( Dot(v[iter],r) );
        r.pexa(v[iter],-a[iter]);
        Orthogonalize(r,v,iter);
        b.push_back( Norm(r) );
        LEigenPair eigen=GSTridiagonal(a.data(),b.data()+1,iter+1);
        lambda0=eigen.eval;
        evec=eigen.evec;
        error=fabs(b[iter+1]*evec[iter]);
        iter++;
    }

    Ket GetState()
    {
        Ket x=v[0]*evec[0];
        for(int i=1;i<iter;i++)
            x.pexa(v[i],evec[i]);
        return x;
    }

    vector<array<Ket,2>> CorrectionV(const vector<cmpx>& ws)  // 1/(z-H) |a> where a=r0
    {
        vector<array<Ket,2>> cvs;

        for(const auto& coeff:CorrectionVTrid(ws,a.data(),b.data()+1,a.size()) )
        {
            Ket x=v[0]-v[0], y=x;
            for(int i=0;i<iter;i++)
            {
                x+=v[i]*coeff[i].real();
                y+=v[i]*coeff[i].imag();
            }
            cvs.push_back({x*b[0],y*b[0]});
        }
        return cvs;
    }

    void DoIt(int nIter, double tol)
    {
//        tol=std::max(tol,nIter*std::numeric_limits<double>::epsilon());
//        double tolr=0;
        for(int i=0;i<nIter;i++)
        {
//            tolr=std::max(tol*fabs(lambda0),tol);
            Iterate();
            if (error<tol) break;
        }
//        if (error>tol)
//            std::cout<<"lanczos failed, residual_norm = "<<error<<std::endl;
    }
};

template<class LinearOperator, class Ket>
Lanczos<LinearOperator,Ket> create_Lanczos(const LinearOperator& A, const Ket& r0)
{
    return Lanczos<LinearOperator,Ket>(A,r0);
}

#include"utils.h"
template<class Hamiltonian, class Ket>                                      //Portal method
EigenSystem0<Ket> Diagonalize(const Hamiltonian& H,const Ket& wf,int nIter,double tol)
{
    Lanczos<Hamiltonian,Ket> lan(H,wf);
    lan.DoIt(nIter, tol);
    return {lan.lambda0,lan.GetState(),lan.iter};
}



//------------------------- linking lapack  -------------------------



inline LEigenPair GSTridiagonal(double *a, double *b, int size, double tol)
{
    const int nEvals=1;
    LEigenPair eigen; eigen.evec.resize(size);
    int M;
    vector<int> ifail(size);
    int info=LAPACKE_dstevx(LAPACK_COL_MAJOR,'V','I', size, a, b,
                   0.0, 0.0,nEvals,nEvals,tol,&M,&eigen.eval,eigen.evec.data(),size,ifail.data());
    if (info!=0)
        throw std::runtime_error("GSTridiagonal: LAPACKE_dstevx info!=0");
//        std::cout<<"GSTridiagonal: LAPACKE_dstevx info!=0\n";
    return eigen;
}

inline array<stdvec,2> EigenFullTridiagonal(double *a,double *b,int n,double tol)
{
    stdvec eval(n), evec(n*n);
    int M;
    vector<int> ifail(n);
    int info=LAPACKE_dstevx(LAPACK_COL_MAJOR,'V','A', n, a, b,
                   0.0, 0.0,0,0,tol,&M,eval.data(),evec.data(),n,ifail.data());
    if (info!=0)
        throw std::runtime_error("GSTridiagonal: LAPACKE_dstevx info!=0");
//        std::cout<<"GSTridiagonal: LAPACKE_dstevx info!=0\n";
    return {eval,evec};
}

inline vector<vector<cmpx>> CorrectionVTrid(const vector<cmpx>& ws,double *a,double *b, int n)
{
    auto eigen=EigenFullTridiagonal(a,b,n);
    auto eval=eigen[0];
    auto evec=eigen[1];
    vector<vector<cmpx>> cvs;
    for (const cmpx& z:ws)
    {
        vector<cmpx> cv(n,0.0);
        for(int i=0;i<n;i++)
            for(int k=0;k<n;k++)
                cv[i]+=evec[i+n*k]*evec[0+n*k]/(z-eval[k]);
        cvs.push_back(cv);
    }
    return cvs;
}


#endif // LANCZOS_H
