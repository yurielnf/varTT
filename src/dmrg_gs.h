#ifndef DMRG_GS_H
#define DMRG_GS_H

#include"superblock.h"
#include"lanczos.h"

struct DMRG_gs
{
    int nIterMax=512,iter;
    double ener,tol_diag=1e-13;
private:
    MPS _gs;
public:
    Superblock sb;

    DMRG_gs(const MPO& mpo,int m)
        :_gs( MPS(mpo.length,m)
              .FillRandu({m,2,m})
              .Canonicalize()
              .Normalize() )
        ,sb({_gs,mpo,_gs})
    {}
    void Solve()
    {
        auto Heff=sb.Oper();
        auto lan=Diagonalize(Heff, sb.mps[0].C, nIterMax, tol_diag);  //Lanczos
        ener=lan.lambda0*sb.Norm();
        iter=lan.iter;
        sb.mps[0].C=sb.mps[2].C=lan.GetState();
    }
    void Print() const
    {
        std::cout<<sb.pos+1<<" "<<sb.length-sb.pos-1;
        std::cout<<" m="<<sb.b1[sb.pos].dim[0]<<" M="<<sb.b1[sb.pos].dim[1]<<" ";
        std::cout<<iter<<" lancz iter; ener="<<ener<<"="<<sb.value()<<"\n";
    }
};

#endif // DMRG_GS_H
