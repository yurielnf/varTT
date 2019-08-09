#ifndef LANCZOS_H
#define LANCZOS_H

#include<iostream>
#include<vector>
#include<stdexcept>

#ifndef MKL
 #include<lapacke.h>
#else
 #include<mkl_lapacke.h>
#endif

using std::vector;

struct LEigenPair
{
    double eval;
    vector<double> evec;
};


template<class Ket>
void Orthogonalize(Ket &t,const std::vector<Ket>& basis,int size)
{
    const double k=0.25;
    double tauin=Norm(t);
    for (int i = 0; i < size; ++i)
        t+=basis[i]*(-Dot(basis[i],t));
    if ( Norm(t)/tauin > k ) return;
    for (int i = 0; i < size; ++i)
        t+=basis[i]*(-Dot(basis[i],t));
}

LEigenPair GSTridiagonal(double *a, double *b, int size, double tol=0);


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
        if (iter>0) r+=v[iter-1]*(-b[iter]);
        a.push_back( Dot(v[iter],r) );
        r+=v[iter]*(-a[iter]);
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
            x+=v[i]*evec[i];
        return x;
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
        if (error>tol)
            std::cout<<"lanczos failed, residual_norm = "<<error<<std::endl;
    }
};

template<class LinearOperator, class Ket>
Lanczos<LinearOperator,Ket> create_Lanczos(const LinearOperator& A, const Ket& r0)
{
    return Lanczos<LinearOperator,Ket>(A,r0);
}

template<class Hamiltonian, class Ket>                                      //Portal method
Lanczos<Hamiltonian,Ket> Diagonalize(const Hamiltonian& H,const Ket& wf,int nIter,double tol)
{
    Lanczos<Hamiltonian,Ket> lan(H,wf);
    lan.DoIt(nIter, tol);
    return lan;
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



#endif // LANCZOS_H
