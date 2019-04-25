#ifndef DMRG_GS_H
#define DMRG_GS_H

#include"superblock.h"
#include"lanczos.h"

struct DMRG_gs
{
    int nIterMax=128,iter;
    double tol_diag=1e-13;
public:
    MPS gs,mpo;
    Superblock sb;

    DMRG_gs(const MPO& ham,int m)
        :mpo(ham)
    {
        gs= MPS(mpo.length,m)
                      .FillRandu({m,2,m})
                      .Canonicalize()
                      .Normalize() ;
        sb=Superblock({&gs,&mpo,&gs});
    }
    void SetPos(MPS::Pos p) { sb.SetPos(p); }
    void Solve()
    {
        auto Heff=sb.Oper();
        auto lan=Diagonalize(Heff, sb.mps[0]->C, nIterMax, tol_diag);
        iter=lan.iter;
        gs.C=lan.GetState();
    }
    void Print() const
    {
        sb.Print();
        std::cout<<"; "<<iter<<" lancz\n";
        std::cout.flush();
    }
};

#endif // DMRG_GS_H
