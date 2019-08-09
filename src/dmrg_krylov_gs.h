#ifndef DMRG_KRYLOV_GS_H
#define DMRG_KRYLOV_GS_H

#include"superblock.h"
#include"lanczos.h"

struct DMRG_krylov_gs
{
    MPO mpo;
    int m, nk, ck, nIterMax=256, iter;
    int nsite_gs=0, nsite_resid=1;
    double tol_diag=1e-13, error=1, errort;

    vector<MPS> gs;
    vector<Superblock> sb_h, sb_o;
    stdvec hmat, omat, eval, evec;

    DMRG_krylov_gs(const MPO& mpo,int m,int n_krylov)
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
        if (i==0)
        {
            int d=mpo.at(0).dim[1];
            if (gs[i].length==0)
                gs[i]= MPS(mpo.length,m)
                        .FillRandu({m,d,m})
                        .Canonicalize()
                        .Normalize();
            sb_h[0]=Superblock({&gs[i],&mpo,&gs[i]});
        }
        else //if (gs[i].length==0)
        {
            gs[i]=ExactRes(i);
            gs[i].Normalize();
            sb_h[i+(i-1)*nk]=Superblock({&gs[i-1],&mpo,&gs[i]});  //reverse list
        }
        for(int j=0;j<i;j++)
                sb_o[i+j*nk]=Superblock({&gs[j],&gs[i]});
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
        MPS x=MPO_MPS{mpo,gs[i-1]}.toMPS(2*m,tol_diag*fabs(hmat[0]));
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
        if (error<tol_diag) return;
        int m=gs[0].m;
//        for(int j=0;j<1;j++)
        {
            MPSSum sum(m,MatSVDFixedDim(m));
            for(int i=1;i<nk;i++)
                sum+=gs[i]*evec[i];
            gs[0].decomposer=MatSVDFixedDim(m);
            gs[0]*=evec[0];
            gs[0]+=sum.toMPS().Sweep();
            gs[0].decomposer=MatQRDecomp;
        }
        ck=0;
        add_MPS();
    }
    void Solve_gs()
    {
        double errord=std::max(error/mpo.length,tol_diag);
        auto lan=Diagonalize(sb_h[0].Oper(nsite_gs), gs[0].CentralMat(nsite_gs), nIterMax, errord);  //Lanczos
        iter=lan.iter;
        gs[0].setCentralMat(lan.GetState());
        gs[0].Normalize();
        if (nsite_gs) sb_h[0].UpdateBlocks();
    }
/*    bool Solve_res()
    {
        int i=ck-1;
        auto beff = sb_h[i+(i-1)*nk].Oper()*gs[i-1].CentralMat();
//        for(int j=0;j<i;j++)
//            beff -= sb_o[i+j*nk].Oper()*gs[j].C*hmat[j+(i-1)*nk];
        bool ok=(Norm(beff)>tol_diag);
        if (!ok) {std::cout<<" small res "; std::cout.flush(); beff=gs[i-1].C;}
        gs[i].setCentralMat(beff);
        gs[i].Normalize();
        return true;
    }*/
/*    bool Solve_res1()
//    {
//        int i=ck-1;
//        auto beff = sb_h[i+(i-1)*nk].Oper(1)*gs[i-1].CentralMat(1);
//        for(int j=0;j<i;j++)
//            beff -= sb_o[i+j*nk].Oper(1)*gs[j].CentralMat(1)*hmat[j+(i-1)*nk];
//        bool ok=(Norm(beff)>tol_diag);
//        if (!ok) {std::cout<<" small res "; std::cout.flush(); beff=gs[i-1].CentralMat(1);}
//        gs[i].setCentralMat( beff );
//        sb_h[i+(i-1)*nk].UpdateBlocks();
//        for(int j=0;j<i;j++)
//            sb_o[i+j*nk].UpdateBlocks();
//        return ok;
//    }
*/
    void Solve_res()
    {
        int i=ck-1;
        auto beff= sb_h[i+(i-1)*nk].Oper(nsite_resid)*gs[i-1].CentralMat(nsite_resid);
//        auto cH=beff;
//        for(int j=0;j<i;j++)
//        {
//            auto cO=sb_o[i+j*nk].Oper(nsite_pert)*gs[j].CentralMat(nsite_pert);
//            beff -= cO*hmat[j+(i-1)*nk];
//        }
        gs[i].setCentralMat( beff );
        gs[i].Normalize();
        if (nsite_resid)
        {
            sb_h[i+(i-1)*nk].UpdateBlocks();
            for(int j=0;j<i;j++)
                sb_o[i+j*nk].UpdateBlocks();
        }
    }
    void SetPos(MPS::Pos p)
    {
        int i=ck-1;
        if (i==0)
            sb_h[0].SetPos(p);
        else
            sb_h[i+(i-1)*nk].SetPos(p);
        for(int j=0;j<i;j++)
                sb_o[i+j*nk].SetPos(p);
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
    void DoIt_res(int nsweep=1)
    {
        hmat[0]=sb_h[0].value();
        omat[0]=1;
        eval={hmat[0]};
        evec={1.0};
        while (ck<nk)
        {
            add_MPS();
            for(int i=0;i<nsweep;i++)
            {
                for(auto p:MPS::SweepPosSec(mpo.length))
                {
                    SetPos(p);
                    Solve_res();
//                    if (mpo.length<10 || (p.i+1) % (mpo.length/10)==0) Print_res();
                }
                CalculateEner();
                Print_res();
            }
            error=errort;
        }
    }
    void CalculateEner()
    {
        int i=ck-1,j=i-1;
        gs[i].Normalize();
/*        hmat[i+j*nk]=hmat[j+i*nk]=sb_h[i+j*nk].value();
//        hmat[i+i*nk]=Superblock({&gs[i],&mpo,&gs[i]}).value();
//        omat[i+i*nk]=1;
//        for(int j=0;j<i;j++)
//        {
//            omat[i+j*nk] = omat[j+i*nk] = sb_o[i+j*nk].value();
//            if(j!=i-1)
//                hmat[i+j*nk]=Superblock({&gs[j],&mpo,&gs[i]}).value();
//        }
*/
        //fuerza bruta
        for(int i=0;i<ck;i++)
            for(j=0;j<=i;j++)
            {
                hmat[i+j*nk]=hmat[j+i*nk]=Superblock({&gs[j],&mpo,&gs[i]}).value();
                omat[i+j*nk]=omat[j+i*nk]=(j<i)?Superblock({&gs[j],&gs[i]}).value():1;
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
    }
    void Print() const
    {
        sb_h[0].Print();
        std::cout<<"; "<<iter<<" lancz\n";
        std::cout.flush();
    }
    void Print_res() const
    {
       std::cout<<"\n"<<ck<<" "<<eval[0]<<"\n"; std::cout.flush();
//        std::cout<<gs[i].pos.i+1<<" ";
    }
};


#endif // DMRG_KRYLOV_GS_H
