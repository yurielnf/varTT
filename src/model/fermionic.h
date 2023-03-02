#ifndef FERMIONIC_H
#define FERMIONIC_H

#include<armadillo>
#include"../mps.h"
#include <map>
#include <array>

using namespace std;

class Fermionic
{

public:
    arma::mat Kmat, Umat;
    std::map<std::array<int,4>, double> Vijkl;
    arma::mat Rot;



    Fermionic(arma::mat const& Kmat_, arma::mat const& Umat_, std::map<std::array<int,4>, double> const& Vijkl_={})
        :Kmat(Kmat_), Umat(Umat_), Vijkl(Vijkl_)
    {}

    Fermionic(arma::mat const& Kmat_, arma::mat const& Umat_,
              arma::mat const& Rot_)
        :Kmat(Rot_.t()*Kmat_*Rot_), Umat(Umat_), Rot(Rot_)
    {}

    int length() const { return Kmat.n_rows; }
    MPO Create(int i) const { return Fermi(i,length(),true); }
    MPO Destroy(int i) const { return Fermi(i,length(),false); }

    MPO CidCj(int i, int j) const { return Create(i)*Destroy(j); }

    MPO Kin(double tol=1e-14) const
    {
        int L=length();
        auto h=MPSSum(10,MatSVDFixedTol(tol));
        // kinetic energy bath
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
        {
            if (fabs(Kmat(i,j))<tol) continue;
            h += Create(i)*Destroy(j)*Kmat(i,j) ;
        }
        return h.toMPS().Sweep();
    }

    MPO Interaction(double tol=1e-14) const
    {
        if (!Rot.empty()) return InteractionRot(tol);

        int L=length();
        auto h=MPSSum(10,MatSVDFixedTol(tol));
        // Uij ni nj
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
        {
            if (fabs(Umat(i,j))<tol) continue;
            h += Create(i)*Destroy(i)*Create(j)*Destroy(j)*Umat(i,j) ;
        }

        for(const auto& it : Vijkl)
        {
            auto pos=it.first; // i, j, k, l
            auto coeff=it.second;
            h += Create(pos[0])*Destroy(pos[1])*Create(pos[2])*Destroy(pos[3]) * coeff;
        }
        return h.toMPS().Sweep();
    }

    MPO InteractionRot(double tol=1e-14) const
    {
        int L=length();
        auto h=MPSSum(10,MatSVDFixedTol(tol));
        // Uij ni nj
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
        {
            if (fabs(Umat(i,j))<tol) continue;

            auto ci=MPSSum(2,MatSVDFixedTol(tol));
            for(int a=0;a<L; a++)
                ci += Destroy(a) * Rot(i,a);

            auto cid=MPSSum(2,MatSVDFixedTol(tol));
            for(int a=0;a<L; a++)
                cid += Create(a) * Rot(i,a);

            auto cj=MPSSum(2,MatSVDFixedTol(tol));
            for(int a=0;a<L; a++)
                cj += Destroy(a) * Rot(j,a);

            auto cjd=MPSSum(2,MatSVDFixedTol(tol));
            for(int a=0;a<L; a++)
                cjd += Create(a) * Rot(j,a);

            h += cid.toMPS() * ci.toMPS() * cjd.toMPS() * cj.toMPS() * Umat(i,j);
        }

        return h.toMPS().Sweep();
    }

    MPO Ham(double tol=1e-14) const
    {
        MPSSum h(10, MatSVDFixedTol(tol));
        h += Kin(tol);
        h += Interaction(tol);
        return h.toMPS().Sweep();
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
};


#endif // FERMIONIC_H
