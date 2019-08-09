#ifndef DMRG_WSE_GS_H
#define DMRG_WSE_GS_H



#include"superblock.h"
#include"lanczos.h"

struct DMRG_wse_gs
{
    int nIterMax=128,iter;
    double alpha=1,tol_diag=1e-13;

    MPS gs,mpo,z2_sym;
    Superblock sb,sb_sym;

    DMRG_wse_gs(const MPO& ham,int m,MPO z2_sym=MPO())
        :mpo(ham),z2_sym(z2_sym)
    {
        int d=mpo.at(0).dim[1];
        gs= MPS(mpo.length,m)
                      .FillRandu({m,d,m})
                      .Canonicalize()
                      .Normalize() ;
        Reset_gs();
    }
    void Reset_gs()
    {
        if(z2_sym.length>0)
        {
            gs.decomposer=MatSVDFixedDim(gs.m);
            gs+=z2_sym*gs;
            gs.Normalize();
            gs.decomposer=MatQRDecomp;
        }
        sb=Superblock({&gs,&mpo,&gs});
        if (z2_sym.length>0)
            sb_sym=Superblock({&gs,&z2_sym,&gs});
    }
    static double AdaptAlpha(double alpha,double Eini,double Eopt,double Etrunc)
    {
        const double epsilon=1e-9;
        double d_opt=Eini-Eopt, f=0.9, r;
        double d_trunc=Etrunc-Eopt;
        if ( fabs(d_opt)<epsilon || fabs(d_trunc)<epsilon )
        {
            if (fabs(d_trunc)>epsilon) f=0.9;
            else f=1.001;
        }
        else
        {
            r=fabs(d_trunc)/fabs(d_opt);
            if (d_trunc<0) f=2*(r+1);
            else if(r<0.05) f=1.2-r;
            else if(r>0.3) f=1.0/(r+0.75);
        }
        f=std::max(0.1,std::min(2.,f));
        alpha*=f;
        alpha=std::max(1e-11,std::min(1e2,alpha));
        return alpha;
    }
    void Solve()
    {
        auto& A=gs.at(gs.pos.i);
        auto& C=gs.C;
        auto& B=gs.at(gs.pos.i+1);
        for(int i=0;i<7;i++)
        {
            double Eini=sb.value();
            auto lan=Diagonalize(sb.Oper(), gs.C, nIterMax, tol_diag);  //Lanczos
            double Eopt=lan.lambda0;
            TensorD P;
            if (sb.pos.vx==1)
            {
                auto M=A*lan.GetState();
                P("kbJI")=M("iaI")*sb.Left(1)("ijk")*sb.mps[1]->CentralMat(1)("jabJ");
                auto AC=M.Decomposition(false,MatSVDFixedDimSE(gs.m,(P*alpha).vec()));
//                auto AC=M.Decomposition(false,gs.decomposer);
                A=AC[0]; C=AC[1];
            }
            else
            {
                auto M=lan.GetState()*B;
                P("ijbK")=M("iaI")*sb.Right(1)("IJK")*sb.mps[1]->CentralMat(1)("jabJ");
                auto CB=M.Decomposition(true,MatSVDFixedDimSE(gs.m,(P*alpha).vec()));
//                auto CB=M.Decomposition(true,gs.decomposer);
                C=CB[0]; B=CB[1];
            }
            sb.UpdateBlocks();
            gs.Normalize();
            double Etrunc=sb.value();
            alpha=AdaptAlpha(alpha,Eini,Eopt,Etrunc);
        }
    }
    void SetPos(MPS::Pos p)
    {
        sb.SetPos(p);
        if (z2_sym.length>0) sb_sym.SetPos(p);
    }
    void Print() const
    {
        std::cout<<sb.pos.i+1<<" "<<sb.length-sb.pos.i-1;
        std::cout<<" m="<<sb.b1[sb.pos.i].dim[0]<<" M="<<sb.b1[sb.pos.i].dim[1]<<" ";
        std::cout<<iter<<" lancz iter; ener="<<sb.value()<<"; alpha="<<alpha;
        std::cout<<"\n";
        std::cout.flush();
    }
};

#endif // DMRG_WSE_GS_H
