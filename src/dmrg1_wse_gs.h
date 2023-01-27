#ifndef DMRG1_WSE_GS_H
#define DMRG1_WSE_GS_H

#include"superblock.h"
#include"lanczos.h"

struct DMRG1_wse_gs
{
    int nIterMax=512,iter;
    double error=1,alpha=1,tol_diag=1e-13;

    MPS gs,mpo,z2_sym;
    Superblock sb,sb_sym;

    DMRG1_wse_gs(const MPO& ham,int m,int m0=-1,MPO z2_sym=MPO())
        :mpo(ham),z2_sym(z2_sym)
    {
        if (m0<0) m0=m;
        int d=mpo.at(0).dim[1];
        gs= MPS(mpo.length,m, MatSVDAdaptative(tol_diag,m))
                      .FillRandu({m0,d,m0})
                      .Canonicalize()
                      .Normalize();
        Reset_gs();
    }
    static double AdaptAlpha(double alpha,double Eini,double Eopt,double Etrunc)
    {
        const double epsilon=1e-9;
        double d_opt=Eini-Eopt, f=1, r;
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
        alpha=std::max(1e-11,std::min(100.,alpha));
        return alpha;
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
    void Solve(bool use_arpack=true)
    {
        double Eini=sb.value();
        double errord=std::max(std::min(1.0,error),tol_diag);
        auto lan=use_arpack?DiagonalizeArn(sb.Oper(1), gs.CentralMat(1), nIterMax, errord):
                            Diagonalize   (sb.Oper(1), gs.CentralMat(1), nIterMax, errord);//Lanczos

        double Eopt=lan.lambda0;
        iter=lan.iter;
//        if (lan.lambda0 > ener) return;
        auto M=lan.GetState();
        auto& A=gs.at(gs.pos.i);
        auto& C=gs.C;
        auto& B=gs.at(gs.pos.i+1);
        TensorD P;
        if (sb.pos.vx==1)
        {
//            P("kbJI")=M("iaI")*sb.Left(1)("ijk")*sb.mps[1]->CentralMat(1)("jabJ");
//            P("iaKJ")=M("kbK")*sb.Left(1)("kji")*sb.mps[1]->CentralMat(1)("jabJ");
            const TensorD &L=sb.Left(1); //kji
            const TensorD &W=sb.mps[1]->CentralMat(1); //"jabJ";
            TensorD a=TensorD({M.dim[0],L.dim[1],W.dim[1],M.dim.back(),W.dim.back()}); //kjaKJ
            for(int j=0;j<a.dim.back();j++)
            {
                auto aj=a.Subtensor(j);
                auto Wj=W.Subtensor(j);
                for(int k=0;k<M.dim.back();k++)
                {
                    auto Mk=M.Subtensor(k);
                    auto akj=aj.Subtensor(k);
                    Mk.MultiplyT(Wj,2,1,akj);
                }
            }
            P=TensorD({L.dim[2],W.dim[1],M.dim[2],W.dim.back()}); //iaKJ
            L.TMultiply(a,2,2,P);

//            P=P.ReShape({1,2}).Clone();
            P*=alpha/Norm(P);
            auto AC=M.Decomposition(false,MatSVDFixedDimSE(gs.m,P.vec(),tol_diag));
            A=AC[0]; C=AC[1];
        }
        else
        {
//            P("ijbK")=M("iaI")*sb.Right(1)("IJK")*sb.mps[1]->CentralMat(1)("jabJ");
//            P("kjaI")=M("kbK")*sb.Right(1)("KJI")*sb.mps[1]->CentralMat(1)("jabJ");
            const TensorD &R=sb.Right(1); //KJI
            const TensorD &W=sb.mps[1]->CentralMat(1);
            TensorD a=M*R; //kbJI
            P=TensorD({a.dim[0],W.dim[0],W.dim[1],R.dim.back()});
            for(int i=0;i<a.dim.back();i++)
            {
                auto ai=a.Subtensor(i);
                auto Pi=P.Subtensor(i);
                ai.MultiplyT(W,2,2,Pi);
            }
//            P=P.ReShape({2,3}).Clone();
            P*=alpha/Norm(P);
            auto CB=M.Decomposition(true,MatSVDFixedDimSE(gs.m,P.vec(),tol_diag));
            C=CB[0]; B=CB[1];
        }
        gs.Normalize();
        sb.UpdateBlocks();
        if (z2_sym.length>0) sb_sym.UpdateBlocks();
        double Etrunc=sb.value();
        error=fabs(Etrunc-Eopt);
        alpha=AdaptAlpha(alpha,Eini,Eopt,Etrunc);
    }

    void SolveNoWSE(bool use_arpack=true)
    {
        double errord=std::max(std::min(1.0,error),tol_diag);
        auto lan=use_arpack?DiagonalizeArn(sb.Oper(1), gs.CentralMat(1), nIterMax, errord):
                            Diagonalize   (sb.Oper(1), gs.CentralMat(1), nIterMax, errord);//Lanczos
        iter=lan.iter;
        gs.setCentralMat(lan.GetState());
        gs.Normalize();
        sb.UpdateBlocks();
        if (sb_sym.length>0) sb_sym.UpdateBlocks();
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

inline void TypicalRunWSE(MPO op,int nsweep,int m)
{
    std::cout<<std::setprecision(15);
    op.PrintSizes("Ham=");
    op.decomposer=MatQRDecomp;  //MatChopDecompFixedTol(0);
    DMRG1_wse_gs sol(op,m);
    sol.tol_diag=1e-12;
    for(int k=0;k<=nsweep;k++)
    {
        for(auto p : MPS::SweepPosSec(op.length))
        {
            sol.SetPos(p);
            sol.Solve();
            if ((p.i+1) % (op.length/10) ==0) sol.Print();
        }
        std::cout<<"sweep "<<k+1<<"\n";
//        if (k>=3) {sol.tol_diag=1e-9; }
//        if (k>=8) {sol.tol_diag=1e-11; }
//        if (k>=12){sol.tol_diag=1e-13; }
    }
}

#endif // DMRG1_WSE_GS_H
