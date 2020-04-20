#ifndef LANCZOSCV_H
#define LANCZOSCV_H

#include"lanczos.h"

template<class LinearOperator, class Ket>
struct LanczosCV
        //Find the lowest eigenvalue lambda0, and eigenvector .GetState()
        //for the eigen-problem A x = lambda x, A Hermitian
{
    const LinearOperator& A;
    Ket r;
    vector<Ket> v;      // Orthonormal basis for the Krylov space
    vector<double> a,b; // the tridiagonal matrix: principal and second diagonals
    int iter;
    vector<cmpx> ws;

    double error=1;

    LanczosCV(const LinearOperator& A, const Ket& r0,vector<cmpx> ws)
        :A(A)
        ,r(r0)
        ,b({Norm(r0)})
        ,iter(0)
        ,ws(ws)
    {}

    double CalculaErrorCV() const
    {
        double norma=0;
        auto cvc=CorrectionVTrid(ws,a.data(),b.data()+1,iter+1);
        for(uint i=0;i<ws.size();i++)
        {
            double eta=ws[i].imag();
            for(int i=0;i<iter;i++)
            {
//                x+=cvc[i].real();
//                y+=cvc[i].imag();
            }
//            cvs.push_back({x*b[0],y*b[0]});
        }
        return norma;
    }

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
        if (error>tol)
            std::cout<<"lanczos failed, residual_norm = "<<error<<std::endl;
    }
};

#endif // LANCZOSCV_H
