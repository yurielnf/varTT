#ifndef DMRG_0_GS_H
#define DMRG_0_GS_H


#include"superblock.h"
#include"lanczos.h"

struct DMRG_0_gs
{
    int nIterMax=128,iter;
    double ener, alpha=1e-3, tol_diag=1e-13,error=1;
public:
    MPS gs,dgs;
    MPO mpo,z2_sym;
    Superblock sb_h11, sb_o12, sb_h12,sb_h22, sb_sym;
    stdvec a,b;

    DMRG_0_gs(const MPO& mpo,int m,int mMax,MPO z2_sym=MPO())
        :mpo(mpo),z2_sym(z2_sym),a(2),b(1)
    {
        int d=mpo.at(0).dim[1];
        auto gs= MPS(mpo.length,mMax)
                      .FillRandu({m,d,m})
                      .Canonicalize() ;
        if(z2_sym.length>0)
        {
            gs.decomposer=MatSVDFixedDim(m);
            gs+=z2_sym*gs;
            gs.Sweep();
        }
        gs.Normalize();
        set_gs(gs);

//        Solve_gs();
//        Solve_res();
//        auto eigen=GSTridiagonal(a.data(),b.data(),2);
//        error=fabs(ener-eigen.eval);
    }
    void set_gs(const MPS& _gs)
    {
        gs=_gs;
        int m=gs.MaxVirtDim();
        int d=mpo.at(0).dim[1];
        gs.decomposer=MatQRDecomp;
        sb_h11=Superblock({&gs,&mpo,&gs});
        if (z2_sym.length>0)
            sb_sym=Superblock({&gs,&z2_sym,&gs});
        ener=sb_h11.value();

        dgs= MPS(mpo.length,m)
                .FillRandu({m,d,m})
                .Canonicalize();
        if(z2_sym.length>0)
        {
            dgs.decomposer=MatSVDFixedDim(m);
            dgs+=z2_sym*dgs;
        }
        dgs.Normalize();
//        sb_h12=Superblock({&gs,&mpo,&dgs});
//        sb_h22=Superblock({&dgs,&mpo,&dgs});
//        sb_o12 =Superblock({&gs,&dgs});
    }
    void reset_gs()
    {
        auto eigen=GSTridiagonal(a.data(),b.data(),2);
        error=fabs(ener-eigen.eval); //fabs(b[0]*eigen.evec[1]);
//        error=tol_diag;
//        if (error==0.0)
//        {
//            sb_h11=Superblock({&gs,&mpo,&gs});
//            if (z2_sym.length>0)
//                sb_sym=Superblock({&gs,&z2_sym,&gs});
//            return;
//        }
        if (fabs(eigen.evec[0])<1e-16)
        {
            eigen.evec[0]=alpha;
            eigen.evec[1]=sqrt(1-eigen.evec[0]*eigen.evec[0]);
            std::cout<<"evec={0,1}\n";
        }
        else if(fabs(eigen.evec[1])<1e-16)
        {
            eigen.evec[1]=alpha;
            eigen.evec[0]=sqrt(1-eigen.evec[1]*eigen.evec[1]);
            std::cout<<"evec={1,0}\n";
        }
        MPSSum gsn(gs.m,MatSVDFixedDim(gs.m));
        gsn+=gs*eigen.evec[0];
        gsn+=dgs*eigen.evec[1];
        std::cout<<"enerL="<<eigen.eval<<"\n";
        set_gs( gsn.toMPS().Normalize() );
    }
    void Solve_gs()
    {
        double errord=std::max(error/10,tol_diag);
        auto lan=Diagonalize(sb_h11.Oper(), gs.C, nIterMax, errord);  //Lanczos
        iter=lan.iter;
        if (lan.lambda0 > ener) return;
        ener=lan.lambda0;
        gs.C=lan.GetState();
        gs.Normalize();
    }
    void Solve_res()
    {
        auto d_psiC= sb_h12.Oper()*gs.C
                    -sb_o12.Oper()*gs.C*ener;
        a[0]=ener;
        b[0]=Norm(d_psiC);
        if (b[0]>error)
            dgs.C=d_psiC*(1.0/b[0]);
        else
        {
            auto phi=sb_o12.Oper()*gs.C;
            dgs.C+=phi*(-Dot(phi,dgs.C));
            double errord=std::max(error/10,tol_diag);
            auto lan=Diagonalize(sb_h22.Oper(), dgs.C, 16, errord);  //Lanczos
            iter=lan.iter;
            dgs.C=lan.GetState();
            dgs.Normalize();
            b[0]=Dot(dgs.C,d_psiC);
        }
        a[1]=sb_h22.value();
    }
    void SetPos_gs(MPS::Pos p)
    {
        sb_h11.SetPos(p);
        if (z2_sym.length>0) z2_sym.SetPos(p);
    }
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
    void Print_res() const
    {
        if (iter==1) {std::cout<<gs.pos.i<<" "; return;}
        std::cout<<" opt expansion ";
        sb_h22.Print();
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
//            SetPos_res(p);
//            Solve_res();
            if ((p.i+1) % (gs.length/10) ==0) Print();
//            if (p.i==gs.length-2 && p.vx==1) reset_gs();
        }
        sb_h12=Superblock({&gs,&mpo,&dgs});
        sb_h22=Superblock({&dgs,&mpo,&dgs});
        sb_o12 =Superblock({&gs,&dgs});
        iter=1;
        for(int i=0;i<2;i++)
        for(auto p:MPS::SweepPosSec(gs.length))
        {
//            SetPos_gs(p);
            SetPos_res(p);
            Solve_res();
            if ((p.i+1) % (gs.length/10) ==0) Print_res();
        }
        reset_gs();
    }
};

#endif // DMRG_0_GS_H
