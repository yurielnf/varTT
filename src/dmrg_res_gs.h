#ifndef DMRG_0_GS_H
#define DMRG_0_GS_H


#include"superblock.h"
#include"lanczos.h"
#include"gmres_m.h"

struct DMRG_0_gs
{
    int nIterMax=256,iter;
    double ener, enerl,alpha=1e-3, tol_diag=1e-13,error=1,errort;
public:
    MPS gs,dgs;
    MPO mpo,z2_sym;
    Superblock sb_h11, sb_o12, sb_h12,sb_h22, sb_sym;
    stdvec a,b;

    DMRG_0_gs(const MPO& mpo,int m,int mMax,double mfactor=1,MPO z2_sym=MPO())
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
                    dgs/*dgsn.toMPS().Normalize()*/ );
    }
    void Solve_gs()
    {
        double errord=std::max(error/gs.length,tol_diag);
        auto lan=Diagonalize(sb_h11.Oper(), gs.C, nIterMax, errord);  //Lanczos
        iter=lan.iter;
        ener=lan.lambda0;
        gs.C=lan.GetState();
        gs.Normalize();
    }
    bool Solve_res()
    {
        auto ch=sb_h12.Oper()*gs.C;
        auto co=sb_o12.Oper()*gs.C; co*=1.0/Norm(co);
        auto beff= ch - co*Dot(co,ch);
//        if (Norm(beff)<error) {return false;}
        dgs.C=beff;
        dgs.Normalize();
        return true;
    }
    bool Solve_res1()
    {
        auto M=gs.CentralMat(1);
        auto ch=sb_h12.Oper(1)*M;
        auto co=sb_o12.Oper(1)*M; co*=1.0/Norm(co);
        auto beff= ch - co*Dot(co,ch);
        if (Norm(beff)<error) {std::cout<<" small res "; std::cout.flush(); beff=M;}
        dgs.setCentralMat( beff );
        dgs.Normalize();
        sb_h12.UpdateBlocks();
        sb_o12.UpdateBlocks();
        return (Norm(beff)>=error);
    }
    void Solve_res1(Superblock& sb_r)
    {
        dgs.setCentralMat( sb_r.Oper(1)*gs.CentralMat(1) );
        dgs.Normalize();
        sb_r.UpdateBlocks();
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
    void Solve_jd1()
    {
        static int c=0;
        auto M=gs.CentralMat(1);
        auto cH=sb_h12.Oper(1)*M;
        auto cO=sb_o12.Oper(1)*M;
        auto beff=cH+cO*(-ener);
        auto A=JDOper(sb_h22.Oper(1),cH,cO,ener,std::min(ener,enerl));
        auto x=dgs.CentralMat(1);
        double errord=std::max(error/2,tol_diag);
        if (c>2*gs.length && Norm(beff)>errord)
        {
//            x.FillZeros();
            auto lan=SolveGMMRES(A,beff,x,nIterMax,errord);
            iter=lan.iter;
            dgs.setCentralMat(lan.x);
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
        c++;
    }
    void Solve_res_opt()
    {
        auto d_psiC= sb_h12.Oper()*gs.C
                    -sb_o12.Oper()*gs.C*ener;
        a[0]=ener;
        {
            auto phi=sb_o12.Oper()*gs.C;
            dgs.C+=phi*(-Dot(phi,dgs.C));
            double errord=std::max(error/2,tol_diag);
            auto lan=Diagonalize(sb_h22.Oper(), dgs.C, nIterMax, errord);  //Lanczos
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
//        sb_h22.SetPos(p);
    }
    void DoIt_gs()
    {
        for(auto p:MPS::SweepPosSec(gs.length))
        {
            SetPos_gs(p);
            Solve_gs();
            if (gs.length<10 || (p.i+1) % (gs.length/10)==0) Print();
        }       
    }
    void DoIt_res(bool opt=false)
    {
        sb_h12=Superblock({&gs,&mpo,&dgs});
        sb_o12 =Superblock({&gs,&dgs});
        iter=1;
//        bool opt=false;
        for(int i=0;i<2;i++)
            for(auto p:MPS::SweepPosSec(gs.length))
            {
                SetPos_gs(p);
                SetPos_res(p);
                if (opt) Solve_res_opt();
                else Solve_res1();
//                if (gs.length<10 || (p.i+1) % (gs.length/10)==0) Print_res();
            }
        sb_h22=Superblock({&dgs,&mpo,&dgs});
        CalculateEner();
        error=errort;
        reset_states();
    }
    void DoIt_resExact()
    {
//        int d=mpo.at(0).dim[1];
//        auto ph=mpo+MPOIdentity(mpo.length,d)*(-ener);
//        ph.Canonicalize();
//        dgs=MPO_MPS{ph,gs}.toMPS(gs.m*2,tol_diag*fabs(ener));
        dgs=MPO_MPS{mpo,gs}.toMPS(gs.m*2,tol_diag*fabs(ener));
//        dgs=mpo*gs;
//        dgs.decomposer=MatSVDAdaptative(tol_diag*fabs(ener),2*gs.m);
//        dgs.Sweep();
        dgs.PrintSizes("dgs");
        dgs.m=gs.m;
        dgs.decomposer=MatSVDFixedDim(gs.m);
        dgs+=gs*(-ener);
        dgs.Normalize();
        dgs.decomposer=MatQRDecomp;
//        auto sb_r=Superblock({&gs,&ph,&dgs});
        sb_h12=Superblock({&gs,&mpo,&dgs});
        sb_o12 =Superblock({&gs,&dgs});
        for(int i=0;i<2;i++)
        for(auto p:MPS::SweepPosSec(gs.length))
        {
            SetPos_gs(p);
            SetPos_res(p);
//            sb_r.SetPos(p);
            Solve_res();
        }
//        sb_h11=Superblock({&gs,&mpo,&gs});        
        sb_h22=Superblock({&dgs,&mpo,&dgs});
        CalculateEner();
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
        enerl=eigen.eval;
//        dgs*=f;
        if (gs.pos.i==0 && gs.pos.vx==-1)
            std::cout<<" o="<<o<<"\n";
    }
    void Print() const
    {
        sb_h11.Print();
        std::cout<<"; "<<iter<<" lancz\n";
        std::cout.flush();
    }
    void Print_res() const
    {
        if (iter==1) {std::cout<<gs.pos.i+1<<" "; /*std::cout.flush();*/ return;}
        std::cout<<" opt expansion ";
        sb_h22.Print();
        std::cout<<"; "<<iter<<" lancz\n";
        std::cout.flush();
    }
};

#endif // DMRG_0_GS_H
