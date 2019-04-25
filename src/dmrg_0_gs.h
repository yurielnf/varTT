#ifndef DMRG_0_GS_H
#define DMRG_0_GS_H


#include"superblock.h"
#include"lanczos.h"

struct DMRG_0_gs
{
    int nIterMax=128,iter;
    double ener,tol_diag=1e-13,error=1;
public:
    MPS gs,dgs;
    MPO mpo;
    Superblock sb_h11, sb_o12, sb_h12,sb_h22;
    stdvec a,b;

    DMRG_0_gs(const MPO& mpo,int m,int mMax)
        :mpo(mpo),a(2),b(1)
    {
        auto gs= MPS(mpo.length,mMax)
                      .FillRandu({m,2,m})
                      .Canonicalize()
                      .Normalize() ;
        set_gs(gs);
    }
    void set_gs(const MPS& _gs)
    {
        gs=_gs;
        gs.decomposer=MatQRDecomp;
        sb_h11=Superblock({&gs,&mpo,&gs});
        ener=sb_h11.value();
        int m=gs.MaxVirtDim();
        dgs= MPS(mpo.length,m)
                .FillRandu({m,2,m})
                .Canonicalize()
                .Normalize();
        sb_h12=Superblock({&gs,&mpo,&dgs});
        sb_h22=Superblock({&dgs,&mpo,&dgs});
        sb_o12 =Superblock({&gs,&dgs});
        Solve_gs();
        Solve_res();
    }
    void reset_gs()
    {
        auto eigen=GSTridiagonal(a.data(),b.data(),2);
        MPSSum gsn(gs.m,MatSVDFixedDim(gs.m));
        gsn+=gs*eigen.evec[0];
        gsn+=dgs*eigen.evec[1];
        std::cout<<"enerL="<<eigen.eval<<"\n";
        set_gs( gsn.toMPS().Canonicalize().Normalize() );
    }
    void Solve_gs()
    {
        auto lan=Diagonalize(sb_h11.Oper(), gs.C, nIterMax, error/gs.length/2);  //Lanczos
        iter=lan.iter;
        if (lan.lambda0 > ener) return;
        ener=lan.lambda0;
        gs.C=lan.GetState();
    }
    void Solve_res()
    {
        auto d_psiC= sb_h12.Oper()*gs.C
                    -sb_o12.Oper()*gs.C*ener;
        a[0]=ener;
        b[0]=Norm(d_psiC); d_psiC*=1.0/b[0];
        dgs.C=d_psiC;
        a[1]=sb_h22.value();
        auto eigen=GSTridiagonal(a.data(),b.data(),2);
        error=std::min(error,fabs(b[0]*eigen.evec[0]));
    }
    void SetPos_gs(MPS::Pos p) { sb_h11.SetPos(p);}
    void SetPos_res(MPS::Pos p)
    {
        sb_o12.SetPos(p);
        sb_h12.SetPos(p);
        sb_h22.SetPos(p);
    }
    void Print() const
    {
        sb_h11.Print();
        std::cout<<"; "<<iter<<" lancz\n";
        std::cout.flush();
    }
    void DoIt()
    {
        auto pos_sec=MPS::SweepPosSec(gs.length);
        for(auto p:pos_sec)
        {
            SetPos_gs(p);
            Solve_gs();
            SetPos_res(p);
            Solve_res();
            if ((p.i+1) % (gs.length/16) ==0) Print();
        }
        for(int i=0;i<1;i++)
        for(auto p:pos_sec)
        {
            SetPos_res(p);
            Solve_res();
        }
        reset_gs();
    }



};

#endif // DMRG_0_GS_H
