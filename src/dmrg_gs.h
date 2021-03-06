#ifndef DMRG_GS_H
#define DMRG_GS_H

#include"superblock.h"
#include"lanczos.h"

struct DMRG_gs
{
    int nIterMax=128,iter;
    double tol_diag=1e-13;
public:
    MPS gs,mpo,z2_sym;
    Superblock sb,sb_sym;

    DMRG_gs(const MPO& ham,int m,MPO z2_sym=MPO())
        :mpo(ham), z2_sym(z2_sym)
    {
        int d=mpo.at(0).dim[1];
        gs= MPS(mpo.length,m)
                      .FillRandu({m,d,m})
                      .Canonicalize()
                      .Normalize() ;
        Reset_gs();
    }
    void Reset_gs()
    {
        if(z2_sym.length>0)
        {
            gs.decomposer=MatSVDFixedDim(gs.m);
            gs+=z2_sym*gs;
            gs.Normalize();
            gs.decomposer=MatQRDecomp;
        }
        sb=Superblock({&gs,&mpo,&gs});
        if (z2_sym.length>0)
            sb_sym=Superblock({&gs,&z2_sym,&gs});
    }

    void SetPos(MPS::Pos p)
    {
        sb.SetPos(p);
        if (z2_sym.length>0) sb_sym.SetPos(p);
    }
    void Solve()
    {
        auto Heff=sb.Oper();
        auto lan=Diagonalize(Heff, gs.C, nIterMax, tol_diag);
        iter=lan.iter;
        gs.C=lan.GetState();        
        gs.Normalize();
    }
    void Print() const
    {
        sb.Print();
        std::cout<<"; "<<iter<<" lancz\n";
        std::cout.flush();
    }
};

#endif // DMRG_GS_H
