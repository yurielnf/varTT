#ifndef DMRG_GS_H
#define DMRG_GS_H

#include"superblock.h"
#include"lanczos.h"

struct DMRG_gs
{
    int nIterMax=128,iter;
    double ener,tol_diag=1e-13;
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
        auto lan=Diagonalize(Heff, sb.mps[0]->C, nIterMax, tol_diag);  //Lanczos
        ener=lan.lambda0;
        iter=lan.iter;
//        sb.mps[0]->C=sb.mps[2]->C=lan.GetState();
        gs.C=lan.GetState();
    }
    void Print() const
    {
        std::cout<<sb.pos.i+1<<" "<<sb.length-sb.pos.i-1;
        std::cout<<" m="<<sb.b1[sb.pos.i].dim[0]<<" M="<<sb.b1[sb.pos.i].dim[1]<<" ";
        std::cout<<iter<<" lancz iter; ener="<<sb.value()<<"\n";
        std::cout.flush();
    }
};

#endif // DMRG_GS_H
