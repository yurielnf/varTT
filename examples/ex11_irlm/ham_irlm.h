#ifndef HAMHALL_H
#define HAMHALL_H

#include<armadillo>
#include"mps.h"
#include"superblock.h"

using namespace std;

class HamIRLM
{

public:
    arma::mat tmat, Pmat;
    double U;
    double tol=1e-10;

    HamIRLM(const char tFile[],const char PFile[], double U_)
        :U(U_)
    {
        tmat.load(tFile);
        Pmat.load(PFile);

    }

    HamIRLM(arma::mat const& tmat_, arma::mat const& Pmat_, double U_)
        :tmat(tmat_), Pmat(Pmat_), U(U_)
    {}

    int length() const { return tmat.n_rows; }
    MPO Create(int i) const { return Fermi(i,length(),true); }
    MPO Destroy(int i) const { return Fermi(i,length(),false); }

    MPO Ham() const
    {
        int L=length();
        auto h=MPSSum(10,MatSVDFixedTol(tol));
        // kinetic energy bath
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
        {
            if (fabs(tmat(i,j))<tol) continue;
            h += Create(i)*Destroy(j)*tmat(i,j) ;
        }

        // interaction
        auto d0=MPSSum(2,MatSVDFixedTol(tol));
        for(int a=0;a<L; a++)
            d0 += Destroy(a) * Pmat(0,a);

        auto d0d=MPSSum(2,MatSVDFixedTol(tol));
        for(int a=0;a<L; a++)
            d0d += Create(a) * Pmat(0,a);

        auto c0=MPSSum(2,MatSVDFixedTol(tol));
        for(int a=0;a<L; a++)
            c0 += Destroy(a) * Pmat(1,a);

        auto c0d=MPSSum(2,MatSVDFixedTol(tol));
        for(int a=0;a<L; a++)
            c0d += Create(a) * Pmat(1,a);

        h += d0d.toMPS() * d0.toMPS() * c0d.toMPS() * c0.toMPS() * U;

        auto H=h.toMPS().Sweep();
        return H;
    }

    MPO NParticle() const
    {
        int L=length();
        int m=4;
        MPSSum npart(m,MatSVDFixedTol(1e-13));
        for(int i=0;i<L; i++)
            npart += Fermi(i,L,true)*Fermi(i,L,false) ;
        return npart.toMPS();
    }

    arma::mat CalculateCiCj(MPS& gs) const
    {
        int N=length();
        arma::mat cicj(N,N);
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++) {
                MPO cc=Fermi(i,N,true)*Fermi(j,N,false);
                cicj(i,j)=Superblock({&gs,&cc,&gs}).value();
            }
        return cicj;
    }
};


#endif // HAMHALL_H
