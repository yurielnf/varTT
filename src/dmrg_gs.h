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
        :_gs( MPS(mpo.length,m).FillRandu({m,2,m}) )
        ,sb({_gs,mpo,_gs})
    {}
    void Solve()
    {
        auto lan=Diagonalize(sb.Oper(), sb.mps[0].C, nIterMax, tol_diag);  //Lanczos
        ener=lan.lambda0;
        iter=lan.iter;
        sb.mps[0].C=lan.GetState();
        Print();
    }
    void Print() const
    {
        std::cout<<sb.pos<<" "<<sb.length-sb.pos;
        std::cout<<" m="<<sb.b1[sb.pos].dim[0]<<" M="<<sb.b1[sb.pos].dim[1]<<" ";
        std::cout<<iter<<" lancz iter; ener="<<ener<<"\n";
    }
};

#endif // DMRG_GS_H
