#ifndef DMRG_0_JD_GS_H
#define DMRG_0_JD_GS_H

#include"superblock.h"
#include"lanczos.h"
#include"gmres_m.h"

struct DMRG_0_JD_gs
{
    int nIterMax=256,iter;
    double ener,enerl, tol_diag=1e-13,error=1;
public:
    MPS gs,dgs;
    MPO mpo,z2_sym;
    Superblock sb_h11, sb_o12, sb_h12,sb_h22, sb_sym;
    stdvec a,b;

    DMRG_0_JD_gs(const MPO& mpo,int m,int mMax,double mfactor=1,MPO z2_sym=MPO())
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
        enerl=ener;

//        Solve_gs();
//        Solve_res();
//        auto eigen=GSTridiagonal(a.data(),b.data(),2);
//        error=fabs(ener-eigen.eval);
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
//        sb_h12=Superblock({&gs,&mpo,&dgs});
//        sb_h22=Superblock({&dgs,&mpo,&dgs});
//        sb_o12 =Superblock({&gs,&dgs});
    }
    void reset_states()
    {
        auto eigen=GSTridiagonal(a.data(),b.data(),2);
        std::cout<<"enerL="<<eigen.eval<<"\n";
        error=fabs(ener-eigen.eval); //fabs(b[0]*eigen.evec[1]);
        enerl=eigen.eval;
//        if (error<tol_diag) {set_states(gs,dgs); return;}
//        error=tol_diag;
//        if (error==0.0)
//        {
//            sb_h11=Superblock({&gs,&mpo,&gs});
//            if (z2_sym.length>0)
//                sb_sym=Superblock({&gs,&z2_sym,&gs});
//            return;
//        }
//        if (fabs(eigen.evec[0])<1e-16)
//        {
//            eigen.evec[0]=alpha;
//            eigen.evec[1]=sqrt(1-eigen.evec[0]*eigen.evec[0]);
//            std::cout<<"evec={0,1}\n";
//        }
//        else if(fabs(eigen.evec[1])<1e-16)
//        {
//            eigen.evec[1]=alpha;
//            eigen.evec[0]=sqrt(1-eigen.evec[1]*eigen.evec[1]);
//            std::cout<<"evec={1,0}\n";
//        }
        double o=sb_o12.value(), nr=1.0/sqrt(1-o*o);
        MPSSum gsn(gs.m,MatSVDFixedDim(gs.m));
        gsn+=gs*(eigen.evec[0]-eigen.evec[1]*nr*o);
        gsn+=dgs*(eigen.evec[1]*nr);

        MPSSum dgsn(dgs.m,MatSVDFixedDim(dgs.m));
        dgsn+=gs*(eigen.evec[1]-eigen.evec[0]*nr*o);
        dgsn+=dgs*(-eigen.evec[0]*nr);

        set_states( gsn .toMPS().Normalize(),
                    dgsn.toMPS().Normalize() );
        set_states( gsn .toMPS().Normalize(), dgs);
    }
    void Solve_gs()
    {
        double errord=std::max(error/gs.length,tol_diag);
        auto lan=Diagonalize(sb_h11.Oper(), gs.C, nIterMax, errord);  //Lanczos
        iter=lan.iter;
//        if (lan.lambda0 > ener) return;
        ener=lan.lambda0;
        gs.C=lan.GetState();
        gs.Normalize();
    }
    struct JDOper
    {
        SuperTensor H22;
        TensorD cH, cO;
        double ener,enerl;
        JDOper(const SuperTensor& H22,const TensorD& cH,const TensorD& cO,
               double ener,double enerl)
            :H22(H22),cH(cH),cO(cO),ener(ener), enerl(enerl) {}
        TensorD operator*(const TensorD& psi) const
        {
            auto y=H22*psi;
            y+=cO*(-Dot(cH,psi));
            y+=cH*(-Dot(cO,psi));
            y+=cO*( Dot(cO,psi)*ener);
            y+=cO*( Dot(cO,psi)*enerl);
            y+=psi*(-enerl);
            return y;
        }
    };

    void Solve_res_opt()
    {
        auto cH=sb_h12.Oper()*gs.C;
        auto cO=sb_o12.Oper()*gs.C;
        auto beff=cH+cO*(-ener);
//        beff*=1.0/Norm(beff);

//        double nr=Norm(dgs.WaveFunction());
//        a[0]=ener;
//        b[0]=sb_h12.value()/nr;
//        a[1]=sb_h22.value()/(nr*nr);
//        auto eigen=GSTridiagonal(a.data(),b.data(),2);
//        std::cout<<" enerl="<<eigen.eval<<" ";
        auto A=JDOper(sb_h22.Oper(),cH,cO,ener,std::min(ener,enerl));
//        double errord=std::max(error/10,tol_diag);
        auto x=dgs.WaveFunction(); //x.FillZeros();
        double errord=std::max(error/gs.length,tol_diag);
        auto lan=SolveGMMRES(A,beff,x,nIterMax,errord);
        iter=lan.iter;
        dgs.SetWaveFunction(lan.x);
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
        std::cout<<" opt expansion ";
        sb_h22.Print();
        std::cout<<"; "<<iter<<" lancz\n";
        std::cout.flush();
    }
    void DoIt_gs()
    {
        for(auto p:MPS::SweepPosSec(gs.length))
        {
            SetPos_gs(p);
            Solve_gs();
            if ((p.i+1)*10 % gs.length ==0) Print();
        }
    }
    void DoIt_res()
    {
        sb_h12=Superblock({&gs,&mpo,&dgs});
        sb_o12 =Superblock({&gs,&dgs});
        sb_h22=Superblock({&dgs,&mpo,&dgs});
        iter=0;
//        nr=Norm(sb_h12.Oper()*gs.C+sb_o12.Oper()*gs.C*(-ener));
        for(int i=0;i<2;i++)
            for(auto p:MPS::SweepPosSec(gs.length))
            {
                //            SetPos_gs(p);
                SetPos_res(p);
                Solve_res_opt();
                if ((p.i+1)*10 % gs.length ==0) Print_res();
            }

//        {
//            MPSSum dgsn(dgs.m);
//            dgsn+=gs*(-sb_o12.value());
//            dgsn+=dgs;
//            dgs=dgsn.toMPS();
//            sb_h12=Superblock({&gs,&mpo,&dgs});
//            sb_o12 =Superblock({&gs,&dgs});
//            sb_h22=Superblock({&dgs,&mpo,&dgs});
//        }

        dgs.Normalize();
        double o=sb_o12.value(), nr=1.0/sqrt(1-o*o);
        a[0]=sb_h11.value();
        b[0]=(sb_h12.value()-a[0]*o)*nr;
        a[1]=(sb_h22.value()-2*o*sb_h12.value()+o*o*a[0])*nr*nr;
//        dgs.decomposer=MatSVDFixedDim(dgs.m); dgs.Sweep();
        reset_states();
    }
};

#endif // DMRG_0_JD_GS_H
