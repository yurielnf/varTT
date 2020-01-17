#ifndef DMRG_JACOBI_DAVIDSON_GS_H
#define DMRG_JACOBI_DAVIDSON_GS_H


#include"superblock.h"
#include"lanczos.h"
#include"gmres.h"

struct DMRG_Jacobi_Davidson_gs
{
    MPO mpo;
    int m, nk, ck, nIterMax=256, iter;
    int nsite_gs=0, nsite_resid=1, nsite_jd=0;
    double tol_diag=1e-13, error=1,beta=1, errort, enerl;

    vector<MPS> gs;
    vector<Superblock> sb_h, sb_o;
    stdvec hmat, omat, eval, evec;

    DMRG_Jacobi_Davidson_gs(const MPO& mpo,int m,int n_krylov)
        :mpo(mpo)
        ,m(m)
        ,nk(n_krylov)
        ,gs(n_krylov)
        ,sb_h(n_krylov * n_krylov)
        ,sb_o(n_krylov * n_krylov)
        ,hmat(n_krylov * n_krylov,0.0)
        ,omat(n_krylov * n_krylov,0.0)
    {
        ck=0;
        add_MPS();
    }
    void add_MPS()
    {
        int i=ck;
        if (i>0)
            gs[i]=ExactRes(i).Normalize();
        else if (gs[i].length==0)
        {
            int d=mpo.at(0).dim[1];
            gs[i]= MPS(mpo.length,m)
                    .FillRandu({m,d,m})
                    .Canonicalize()
                    .Normalize();
        }
        for(int j=0;j<=i;j++)
        {
            sb_h[i+j*nk]=Superblock({&gs[j],&mpo,&gs[i]});  //reverse list
            if (j<i) sb_o[i+j*nk]=Superblock({&gs[j],&gs[i]});
        }
        ck++;
    }

/*    MPS ExactRes(int i)
    {
        MPSSum sum(m,MatSVDFixedDim(m));
        for(int j=0;j<i;j++)
        {
            MPS x=MPO_MPS{mpo,gs[j]}.toMPS(2*m,tol_diag*fabs(eval[0]));
            x.PrintSizes("residual");
            sum+= x*evec[j];
        }
        for(int j=0;j<i;j++)
            sum += gs[j] * ( -evec[j]*eval[0] );

        MPS x=sum.toMPS();
        x.decomposer=MatQRDecomp;
        return x;
    }*/
    MPS ExactRes(int i)
    {
        double errord=std::max(error,tol_diag)*fabs(hmat[0]);
        MPS x=MPO_MPS{mpo,gs[i-1]}.toMPS(2*m,errord);
        //MPS x=mpo*gs[i-1];
        x.PrintSizes("residual");
        MPSSum sum(m,MatSVDFixedDim(m));
        for(int j=0;j<i;j++)
            sum += gs[j] * ( -hmat[j+(i-1)*nk] );
        x.m=m;
        x.decomposer=MatSVDFixedDim(m);
        x+=sum.toMPS();
//        x.Sweep();
        x.decomposer=MatQRDecomp;
        return x;
    }
    void reset_states()
    {
        if (error<tol_diag) {ck=1; return;}
        int m=gs[0].m;
        vector<MPS> gsn(nk);
        for(int j=0;j<1;j++)
        {
            MPSSum sum(m,MatSVDFixedDim(m));
            for(int i=0;i<nk;i++)
                sum+=gs[i]*evec[i+j*nk];
            gsn[j]=sum.toMPS().Sweep(1).Normalize();
            if (j==0) gsn[j].Sweep(1).Normalize();
            gsn[j].decomposer=MatQRDecomp;
        }
        for(int i=0;i<1;i++) gs[i]=gsn[i];
        ck=0;
        add_MPS();
    }

    void Solve_gs()
    {
        double errord=std::max(error/mpo.length,tol_diag);
        auto lan=Diagonalize(sb_h[0].Oper(nsite_gs), gs[0].CentralMat(nsite_gs), nIterMax, errord);  //Lanczos
        iter=lan.iter;
        gs[0].setCentralMat ( lan.GetState() );
        gs[0].Normalize();
        if (nsite_gs) sb_h[0].UpdateBlocks();
    }
    void Solve_res()
    {
        int i=ck-1;
//        TensorD cO,cH;
//        for(int j=0;j<i;j++)
//        {
//            cO+=sb_o[i+j*nk].Oper(nsite_resid)*gs[j].CentralMat(nsite_resid)*evec[j];
//            cH+=sb_h[i+j*nk].Oper(nsite_resid)*gs[j].CentralMat(nsite_resid)*evec[j];
//        }
//        auto beff= cH+cO*(-eval[0]);
        auto beff= sb_h[i+(i-1)*nk].Oper(nsite_resid)*gs[i-1].CentralMat(nsite_resid);
        gs[i].setCentralMat( beff );
        gs[i].Normalize();
        if (nsite_resid)
        {
            sb_h[i+(i-1)*nk].UpdateBlocks();
            for(int j=0;j<i;j++)
                sb_o[i+j*nk].UpdateBlocks();
        }
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
    void Solve_jd()
    {
        int i=ck-1;
        TensorD cO,cH;
        for(int j=0;j<i;j++)
        {
            cO+=sb_o[i+j*nk].Oper(nsite_jd)*gs[j].CentralMat(nsite_jd)*evec[j];
            cH+=sb_h[i+j*nk].Oper(nsite_jd)*gs[j].CentralMat(nsite_jd)*evec[j];
        }
        auto beff= cH+cO*(-eval[0]);
        if (fabs(beta)>tol_diag) beff*=1.0/beta;

        auto A=JDOper(sb_h[i+i*nk].Oper(nsite_jd),cH,cO,eval[0],enerl);
        auto x=gs[i].CentralMat(nsite_jd);
        double errord=std::max(error,tol_diag);
        errord=std::min(errord,1e-2);
        //int nIterMax=20;
        iter=nIterMax;
        //        CG(A,x,beff,iter,errord);
        GMRES(A,x,beff,nIterMax,iter,errord);
        gs[i].setCentralMat( x );
        gs[i].Normalize();
        if (nsite_jd)
        {
            for(int j=0;j<i;j++)
            {
                sb_o[i+j*nk].UpdateBlocks();
                sb_h[i+j*nk].UpdateBlocks();
            }
            sb_h[i+i*nk].UpdateBlocks();
        }
    }

    struct JDFOper
    {
        SuperTensor H22;
        double enerl;
        JDFOper(const SuperTensor& H22,double enerl)
            :H22(H22), enerl(enerl) {}
        TensorD operator*(const TensorD& psi) const
        {
            auto y=H22*psi;
            y+=psi*(-enerl);
            return y;
        }
    };
    void Solve_jdf(bool only_resid=false)
    {
        int i=ck-1;
        TensorD cO,cH;
        for(int j=0;j<i;j++)
        {
            cO+=sb_o[i+j*nk].Oper(nsite_resid)*gs[j].CentralMat(nsite_resid)*evec[j];
            cH+=sb_h[i+j*nk].Oper(nsite_resid)*gs[j].CentralMat(nsite_resid)*evec[j];
        }
        auto beff= cH+cO*(-enerl);
        CurrentEner();
        if (only_resid)
        {
            iter=1;
            gs[i].setCentralMat( beff );
        }
        else
        {
            auto A=JDFOper(sb_h[i+i*nk].Oper(nsite_resid),enerl);
            auto x=gs[i].CentralMat(nsite_resid);
            double errord=std::max(error/(2*mpo.length),tol_diag);
            errord=std::min(errord,1e-2);
            iter=nIterMax;
            //        CG(A,x,beff,iter,errord);
            GMRES(A,x,beff,nIterMax,iter,errord);
            gs[i].setCentralMat( x );
        }

        gs[i].Normalize();
        if (nsite_resid)
        {
            for(int j=0;j<i;j++)
            {
                sb_o[i+j*nk].UpdateBlocks();
                sb_h[i+j*nk].UpdateBlocks();
            }
            sb_h[i+i*nk].UpdateBlocks();
        }
    }
    void SetPos(MPS::Pos p)
    {
        int i=ck-1;
        for(int j=0;j<i;j++)
        {
                sb_o[i+j*nk].SetPos(p);
                sb_h[i+j*nk].SetPos(p);
        }
        sb_h[i+i*nk].SetPos(p);
    }
    void DoIt_gs()
    {
        for(auto p:MPS::SweepPosSec(mpo.length))
        {
            SetPos(p);
            Solve_gs();
            if (mpo.length<10 || (p.i+1) % (mpo.length/10)==0) Print();
        }
    }
    void DoIt_res(int nsweep_resid=1,int nsweep_jd=1)
    {
        hmat[0]=sb_h[0].value();
        omat[0]=1;
        eval={hmat[0]};
        evec={1.0};
        while (ck<nk)
        {
            add_MPS();
            for(int i=0;i<nsweep_resid+nsweep_jd;i++)
            {
                for(auto p:MPS::SweepPosSec(mpo.length))
                {
                    SetPos(p);                    
                    CurrentEner();
                    if (i<nsweep_resid)
                        Solve_res();
                    else
                    {
                        Solve_jd();
                        if (mpo.length<10 || (p.i+1) % (mpo.length/10)==0) Print();
                    }
                }
                Print_res();
            }
            UpdateSB();
            CalculateEner();
            Print_res();
            error=errort;
        }
    }
    void UpdateSB()
    {
        //fuerza bruta
        for(int i=0;i<ck-1;i++)
            for(int j=0;j<=i;j++)
            {
                sb_h[i+j*nk]=Superblock({&gs[j],&mpo,&gs[i]});
                if (j<i)
                    sb_o[i+j*nk]=Superblock({&gs[j],&gs[i]});
                hmat[i+j*nk]=hmat[j+i*nk]=sb_h[i+j*nk].value();
                omat[i+j*nk]=omat[j+i*nk]=(j<i)?sb_o[i+j*nk].value():1.0;
            }
    }

    void CalculateEner()
    {
        int i=ck-1;
        for(int j=0;j<=i;j++)
        {
            hmat[i+j*nk]=hmat[j+i*nk]=sb_h[i+j*nk].value();
            omat[i+j*nk]=omat[j+i*nk]=(j<i)?sb_o[i+j*nk].value():1;
        }

        stdvec hm(ck*ck), om(ck*ck);
        for(int i=0;i<ck;i++)    //cut the matrices from nk*nk to ck*ck
            for(int j=0;j<ck;j++)
            {
                hm[i+j*ck]=hmat[i+j*nk];
                om[i+j*ck]=omat[i+j*nk];
            }
        for(auto x:om) std::cout<<x<<" ";
        std::cout<<"\n";
        eval.resize(ck);
        evec.resize(ck*ck);
        MatFullDiagGen(hm.data(), om.data(), ck, evec.data(), eval.data());
        errort=fabs(hmat[0]-eval[0]);//fabs(hmat[ck-2+(ck-1)*nk]*evec[ck-1]);
        beta=evec[0+ck*(ck-1)];
    }
    double CurrentEner()
    {
        int i=ck-1;
        for(int j=0;j<=i;j++)
        {
            hmat[i+j*nk]=hmat[j+i*nk]=sb_h[i+j*nk].value();
            omat[i+j*nk]=omat[j+i*nk]=(j<i)?sb_o[i+j*nk].value():1;
        }

        stdvec hm(ck*ck), om(ck*ck);
        for(int i=0;i<ck;i++)    //cut the matrices from nk*nk to ck*ck
            for(int j=0;j<ck;j++)
            {
                hm[i+j*ck]=hmat[i+j*nk];
                om[i+j*ck]=omat[i+j*nk];
            }
        stdvec eval(ck);
        stdvec evec(ck*ck);
        MatFullDiagGen(hm.data(), om.data(), ck, evec.data(), eval.data());
        errort=fabs(hmat[0]-eval[0]);//fabs(hmat[ck-2+(ck-1)*nk]*evec[ck-1]);
        beta=evec[0+ck*(ck-1)];
        enerl=eval[0];
        return eval[0];
    }
    void Print() const
    {
        int i=ck-1;
        if (i>0) std::cout<<"-----> Ritz_"<<i<<"; enerl="<<enerl<<"; ";
        sb_h[i+i*nk].Print();
        std::cout<<"; "<<iter<<" krylov\n";
        std::cout.flush();
    }
    void Print_res() const
    {
       std::cout<<"\n"<<ck<<" "<<enerl<<"\n"; std::cout.flush();
//        std::cout<<gs[i].pos.i+1<<" ";
    }
};

#endif // DMRG_JACOBI_DAVIDSON_GS_H
