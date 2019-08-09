#ifndef DMRG1_JD_GS_H
#define DMRG1_JD_GS_H

#include"superblock.h"
#include"lanczos.h"
#include"gmres_m.h"
#include"cg.h"
#include"gmres.h"

struct DMRG1_JD_gs
{
    int nIterMax=128,iter;
    double ener,enerl, tol_diag=1e-13,error=1,errort;
public:
    MPS gs,dgs;
    MPO mpo,z2_sym;
    Superblock sb_h11, sb_o12, sb_h12,sb_h22, sb_sym;
    stdvec a,b;

    DMRG1_JD_gs(const MPO& mpo,int m,int mMax,MPO z2_sym=MPO())
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

        set_states(gs);
        enerl=ener;

    }
    MPS ExactRes()
    {
        int m=gs.m;
        MPS x=MPO_MPS{mpo,gs}.toMPS(2*m,tol_diag*fabs(ener));
        x.PrintSizes("residual");
        x.m=m;
        x.decomposer=MatSVDFixedDim(m);
        x+=gs * ( -ener );
        x.decomposer=MatQRDecomp;
        return x;
    }
    void set_states(const MPS& _gs)
    {
        gs=_gs;
        gs.decomposer=MatQRDecomp;
        sb_h11=Superblock({&gs,&mpo,&gs});
        if (z2_sym.length>0)
            sb_sym=Superblock({&gs,&z2_sym,&gs});
        ener=sb_h11.value();
    }
    void reset_states()
    {
        if (error<tol_diag) {return;}
//        dgs.Normalize();
        double o=sb_o12.value(), nr=1.0/sqrt(1-o*o);
        MPSSum gsn(gs.m,MatSVDFixedDim(gs.m));
        auto eigen=GSTridiagonal(a.data(),b.data(),2);
        gsn+=gs*(eigen.evec[0]-eigen.evec[1]*nr*o);
        gsn+=dgs*(eigen.evec[1]*nr);

//        MPSSum dgsn(dgs.m,MatSVDFixedDim(dgs.m));
//        dgsn+=gs*(eigen.evec[1]-eigen.evec[0]*nr*o);
//        dgsn+=dgs*(-eigen.evec[0]*nr);
//        dgs=dgsn.toMPS().Sweep().Normalize();
        set_states( gsn .toMPS().Sweep().Normalize() );
    }
    void Solve_gs()
    {
        double errord=std::max(error/2,tol_diag);
        auto lan=Diagonalize(sb_h11.Oper(), gs.C, nIterMax, errord);  //Lanczos
        iter=lan.iter;
        ener=lan.lambda0;
        gs.C=lan.GetState();
//        gs.setCentralMat(lan.GetState());
//        sb_h11.UpdateBlocks();
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
        const auto& M=gs.CentralMat(1);
        auto cH=sb_h12.Oper(1)*M;
        auto cO=sb_o12.Oper(1)*M;
        auto beff=cH+cO*(-ener);
        auto A=JDOper(sb_h22.Oper(1),cH,cO,ener,std::min(ener,enerl));
        auto x=dgs.CentralMat(1);
        double errord=std::max(error/(2*gs.length),tol_diag);
        errord=std::min(errord,1e-2);
        if (Norm(beff)>errord)
        {
//            x.FillZeros();
            iter=nIterMax;
//            CG(A,x,beff,iter,errord);
            GMRES(A,x,beff,nIterMax,iter,errord);
//            auto lan=SolveGMMRES(A,beff,x,nIterMax,errord);
//            iter+=lan.iter;
//            x=lan.x;
            dgs.setCentralMat(x);
        }
        else
        {
            std::cout<<" first excited ";
            auto lan=Diagonalize(A,x,nIterMax,errord);
            iter=lan.iter;
            dgs.setCentralMat(lan.GetState());
        }
        sb_o12.UpdateBlocks();
        sb_h12.UpdateBlocks();
        sb_h22.UpdateBlocks();
    }
    void Solve_jd()
    {
        const auto& M=gs.CentralMat();
        auto cH=sb_h12.Oper()*M;
        auto cO=sb_o12.Oper()*M;
        auto beff=cH+cO*(-ener);
        auto A=JDOper(sb_h22.Oper(),cH,cO,ener,std::min(ener,enerl));
        auto x=dgs.CentralMat();
        double errord=std::max(error/2,tol_diag);
        errord=std::min(errord,1e-2);
        if (Norm(beff)>tol_diag)
        {
            x.FillZeros();
//            auto lan=SolveGMMRES(A,beff,x,nIterMax,errord);
//            iter=lan.iter;
            iter=nIterMax;
            CG(A,x,beff,iter,errord);
            dgs.setCentralMat(x);
        }
        else
        {
            std::cout<<" first excited ";
//            auto lan=Diagonalize(A,x,nIterMax,errord);
//            iter=lan.iter;
//            dgs.setCentralMat(lan.GetState());
        }
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
            if ((p.i+1)*10 % gs.length ==0) Print();
        }
    }
    void DoIt_res(int nsweep=1)
    {
        /*if (dgs.length==0) */{dgs=ExactRes(); dgs.Normalize(); }
        sb_h12=Superblock({&gs,&mpo,&dgs});
        sb_o12 =Superblock({&gs,&dgs});
        sb_h22=Superblock({&dgs,&mpo,&dgs});
        iter=0;
        for(int i=0;i<nsweep;i++)
            for(auto p:MPS::SweepPosSec(gs.length))
            {
                SetPos_gs(p);
                SetPos_res(p);
                Solve_res_opt();
                CalculateEner();
                if ((p.i+1)*10 % gs.length ==0) Print_res();
            }
        error=errort;
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
//        std::cout<<" enerL="<<eigen.eval<<"\n";
        errort=fabs(ener-eigen.eval); //fabs(b[0]*eigen.evec[1]);
        enerl=eigen.eval;
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
        std::cout<<" opt expansion ";
        sb_h22.Print();
        std::cout<<"; "<<iter<<" CG; enerl="<<enerl<<"\n";
        std::cout.flush();
    }
};

#endif // DMRG1_JD_GS_H
