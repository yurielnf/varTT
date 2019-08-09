#ifndef DMRG1_RES_GS_H
#define DMRG1_RES_GS_H


#include"superblock.h"
#include"lanczos.h"

struct DMRG1_opt_gs
{
    int nIterMax=128,iter;
    double ener, alpha=1e-3, tol_diag=1e-13,error=1,errort;
public:
    MPS gs,dgs;
    MPO mpo,z2_sym;
    Superblock sb_h11, sb_o12, sb_h12,sb_h22, sb_sym;
    stdvec a,b;

    DMRG1_opt_gs(const MPO& mpo,int m,int mMax,double mfactor=1,MPO z2_sym=MPO())
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

        int mb=int(m*mfactor);
        auto dgs= MPS(mpo.length,mb)
                .FillRandu({mb,d,mb})
                .Canonicalize();
        if(z2_sym.length>0)
        {
            dgs.decomposer=MatSVDFixedDim(m);
            dgs+=z2_sym*dgs;
        }
        dgs.Normalize();
        set_states(gs,dgs);

    }
    void set_states(const MPS& _gs,const MPS& _dgs)
    {
        gs=_gs;
        gs.decomposer=MatQRDecomp;
        sb_h11=Superblock({&gs,&mpo,&gs});
        if (z2_sym.length>0)
            sb_sym=Superblock({&gs,&z2_sym,&gs});
        ener=sb_h11.value();
        dgs=_dgs;
    }
    void reset_states()
    {
        if (error<tol_diag) {set_states(gs,dgs); return;}
//        dgs.Normalize();
        double o=sb_o12.value(), nr=1.0/sqrt(1-o*o);
        MPSSum gsn(gs.m,MatSVDFixedDim(gs.m));
        auto eigen=GSTridiagonal(a.data(),b.data(),2);
        gsn+=gs*(eigen.evec[0]-eigen.evec[1]*nr*o);
        gsn+=dgs*(eigen.evec[1]*nr);

        MPSSum dgsn(dgs.m,MatSVDFixedDim(dgs.m));
        dgsn+=gs*(eigen.evec[1]-eigen.evec[0]*nr*o);
        dgsn+=dgs*(-eigen.evec[0]*nr);

        set_states( gsn .toMPS().Sweep().Normalize(),
                    dgsn.toMPS().Normalize() );
    }
    void Solve_gs()
    {
        double errord=std::max(error/2,tol_diag);
        auto lan=Diagonalize(sb_h11.Oper(1), gs.CentralMat(1), nIterMax, errord);  //Lanczos
        iter=lan.iter;
        ener=lan.lambda0;
        gs.setCentralMat(lan.GetState());
        sb_h11.UpdateBlocks();
        gs.Normalize();
    }
    bool Solve_res()
    {
        const auto& M=gs.CentralMat(1);
        auto beff= sb_h12.Oper(1)*M - sb_o12.Oper(1)*M*ener;
//        if (Norm(beff)<error) {return false;}
        dgs.setCentralMat( beff );
        sb_h12.UpdateBlocks();
        sb_o12.UpdateBlocks();
        sb_h22.UpdateBlocks();
        return true;
    }
    void Solve_res_opt()
    {
        auto d_psiC= sb_h12.Oper(1)*gs.CentralMat(1)
                    -sb_o12.Oper(1)*gs.CentralMat(1)*ener;
        a[0]=ener;
        {
            auto phi=sb_o12.Oper(1)*gs.CentralMat(1);
            //dgs.C+=phi*(-Dot(phi,dgs.C));
            dgs.setCentralMat( dgs.CentralMat(1)
                              +phi*(-Dot(phi,dgs.CentralMat(1))) );
            double errord=std::max(error/10,tol_diag);
            auto lan=Diagonalize(sb_h22.Oper(1), dgs.CentralMat(1), nIterMax, errord);  //Lanczos
            iter=lan.iter;
            dgs.setCentralMat(lan.GetState());
            dgs.Normalize();
            sb_h12.UpdateBlocks();
            sb_o12.UpdateBlocks();
            sb_h22.UpdateBlocks();
            b[0]=Dot(dgs.CentralMat(1),d_psiC);
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
    void DoIt_gs()
    {
        for(auto p:MPS::SweepPosSec(gs.length))
        {
            SetPos_gs(p);
            Solve_gs();
            if ((p.i+1) % (gs.length/10) ==0) Print();
        }        
    }
    void DoIt_res()
    {
        sb_h12=Superblock({&gs,&mpo,&dgs});
        sb_o12 =Superblock({&gs,&dgs});
        sb_h22=Superblock({&dgs,&mpo,&dgs});
        iter=1;
        for(int i=0;i<2;i++)
            for(auto p:MPS::SweepPosSec(gs.length))
            {
                SetPos_gs(p);
                SetPos_res(p);
                Solve_res();
                CalculateEner();
                if ((p.i+1)*10 % gs.length ==0) Print_res();
            }
        error=errort;
        reset_states();
    }
    void CalculateEner()
    {
        gs.Normalize();
//        double f=dgs.norm();
        dgs.Normalize();
        double o=sb_o12.value(), nr=1.0/sqrt(1-o*o);
        a[0]=ener=sb_h11.value();
        b[0]=(sb_h12.value()-a[0]*o)*nr;
        a[1]=(sb_h22.value()-2*o*sb_h12.value()+o*o*a[0])*nr*nr;
        auto eigen=GSTridiagonal(a.data(),b.data(),2);
        std::cout<<" enerL="<<eigen.eval<<"\n";
        errort=fabs(ener-eigen.eval); //fabs(b[0]*eigen.evec[1]);
//        dgs*=f;
    }
    void Print() const
    {
        sb_h11.Print();
        std::cout<<"; "<<iter<<" lancz\n";
        std::cout.flush();
    }
    void Print_res() const
    {
        if (iter==1) {std::cout<<gs.pos.i+1<<" "; return;}
        std::cout<<" opt expansion ";
        sb_h22.Print();
        std::cout<<"; "<<iter<<" lancz\n";
        std::cout.flush();
    }
};


#endif // DMRG1_RES_GS_H
