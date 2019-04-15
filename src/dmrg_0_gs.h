#ifndef DMRG_0_GS_H
#define DMRG_0_GS_H


#include"superblock.h"
#include"lanczos.h"

struct DMRG_0_gs
{
    int nIterMax=128,iter;
    double tol_diag=1e-13;
public:
    MPS gs,dgs;
    MPO mpo;
    Superblock sb_h11, sb_o12, sb_h12,sb_h22;
    stdvec a,b;

    DMRG_0_gs(const MPO& mpo,int m)
        :mpo(mpo),a(2),b(1)
    {
        auto gs= MPS(mpo.length,m)
                      .FillRandu({m,2,m})
                      .Canonicalize()
                      .Normalize() ;
        set_gs(gs);
    }
    void set_gs(const MPS& _gs)
    {
        gs=_gs; gs.decomposer=MatQRDecomp;
        sb_h11=Superblock({&gs,&mpo,&gs});
        int m=gs.m;
        dgs= MPS(mpo.length,m)
                      .FillRandu({m,2,m});
        dgs.Canonicalize().Normalize();
        sb_h12=Superblock({&gs,&mpo,&dgs});
        sb_h22=Superblock({&dgs,&mpo,&dgs});
        sb_o12 =Superblock({&gs,&dgs});
    }
    void reset_gs()
    {
        auto eigen=GSTridiagonal(a.data(),b.data(),2);
        MPSSum gsn(gs.m);
        gsn+=gs*eigen.evec[0];
        gsn+=dgs*eigen.evec[1];
        std::cout<<"enerL="<<eigen.eval<<"\n";
        set_gs( gsn.toMPS().Canonicalize().Normalize() );
    }
    void Solve()
    {
        auto lan=Diagonalize(sb_h11.Oper(), gs.C, nIterMax, tol_diag);  //Lanczos
        iter=lan.iter;
        double ener=lan.lambda0;
        gs.C=lan.GetState();
        auto d_psiC= sb_h12.Oper()*gs.C
                    -sb_o12.Oper()*gs.C*ener;
        a[0]=ener;
        b[0]=Norm(d_psiC); d_psiC*=1.0/b[0];
        dgs.C=d_psiC;
        a[1]=sb_h22.value();
    }

    void SetPos(MPS::Pos p)
    {
        sb_h11.SetPos(p);
        sb_o12.SetPos(p);
        sb_h12.SetPos(p);
        sb_h22.SetPos(p);
    }

    void Print() const
    {
        const auto& sb=sb_h11;
        std::cout<<sb.pos.i+1<<" "<<sb.length-sb.pos.i-1;
        std::cout<<" m="<<sb.b1[sb.pos.i].dim[0]<<" M="<<sb.b1[sb.pos.i].dim[1]<<" ";
        std::cout<<iter<<" lancz iter; ener="<<sb.value();
//        std::cout<<"; h12="<<sb_h12.value()/sb_o12.mps[1].norm_factor();
//        std::cout<<"; h22="<<sb_h22.value()/pow(sb_h22.mps[0].norm_factor(),2);
        std::cout<<"\n";
        std::cout.flush();
    }
};

#endif // DMRG_0_GS_H
