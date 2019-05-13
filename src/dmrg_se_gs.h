#ifndef DMRG_SE_GS_H
#define DMRG_SE_GS_H


#include"superblock.h"
#include"lanczos.h"

struct DMRG_se_gs
{
    int nIterMax=128,iter;
    double ener,tol_diag=1e-13;

    MPS gs,mpo;
    Superblock sb;
    stdvec a,b;

    DMRG_se_gs(const MPO& ham,int m)
        :mpo(ham),a(2),b(1)
    {
        int d=mpo.at(0).dim[1];
        gs= MPS(mpo.length,m)
                      .FillRandu({m,d,m})
                      .Canonicalize()
                      .Normalize() ;
        sb=Superblock({&gs,&mpo,&gs});
        ener=sb.value();
    }
    void Solve()
    {
        auto lan=Diagonalize(sb.Oper(), gs.C, nIterMax, tol_diag);  //Lanczos
        iter=lan.iter;
//        if (lan.lambda0 > ener) return;
        ener=lan.lambda0;
        gs.C=lan.GetState();
        gs.Normalize();

        if (sb.pos.vx==1)
        {
            auto &A=gs.at(gs.pos.i);
            auto &C=gs.C;
            std::array<TensorD,2> AC={A,C};
            auto M=A*C;
            auto Mb= sb.Oper(1)*M - M*ener;
            a[0]=ener;
            b[0]=Norm(Mb);
            if (b[0]<1e-13) return;
            Mb*=1.0/b[0];
            auto ACb=Mb.Decomposition(false,MatQRDecomp);
            A=ACb[0]; C=ACb[1];  //<-------------------  setCentralMat1(Mb)
            sb.UpdateBlocks();
            a[1]=sb.value();
            A=AC[0]; C=AC[1];    //<-------------------  setCentralMat1(M)
            auto eigen=GSTridiagonal(a.data(),b.data(),2);
            A=DirectSum(AC[0]*eigen.evec[0],
                       ACb[0]*eigen.evec[1], true);
            C=DirectSum(AC[1],
                       ACb[1], false);
            sb.UpdateBlocks();
            AC=A.Decomposition(false,MatQRDecomp/*MatSVDFixedDim(gs.m)*/);
            A=AC[0]; C=AC[1]*C;
            sb.UpdateBlocks();
            double ener2=sb.value();
//            double ener3=sb.value1();
            std::cout<<"enerL="<<eigen.eval<<" "<<ener2<<"\n";
        }
    }
    void SetPos(MPS::Pos p) { sb.SetPos(p); }
    void Print() const
    {
        std::cout<<sb.pos.i+1<<" "<<sb.length-sb.pos.i-1;
        std::cout<<" m="<<sb.b1[sb.pos.i].dim[0]<<" M="<<sb.b1[sb.pos.i].dim[1]<<" ";
        std::cout<<iter<<" lancz iter; ener="<<sb.value();
        std::cout<<"\n";
        std::cout.flush();
    }
};

#endif // DMRG_SE_GS_H
