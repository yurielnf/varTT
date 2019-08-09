#ifndef DMRG_RES_CV_H
#define DMRG_RES_CV_H


#include"superblock.h"
#include"correctionvector.h"

struct DMRG_0_cv
{
    int n, nIterMax=2048,iter;
    int nsite=1;
    MPO mpo;
    MPS a;
    double ener;
    stdvec wR,wI;
    MPS cvy, cvy2;
    vector<TensorD> cvR, cvI;
    Superblock sb,sb_ya;
private:
    TensorD rot;
public:

    DMRG_0_cv(const MPO& _mpo,int m,const MPS& _a,double ener,stdvec wR,stdvec wI)
        :n(wR.size())
        ,mpo(_mpo)
        ,a(_a)
        ,ener(ener)
        ,wR(wR),wI(wI)
        ,cvR(n), cvI(n)
    {
        //        int d=mpo.at(0).dim[1];
        cvy= a;/*MPS(mpo.length,m)
                .FillRandu({m,d,m})
                .Canonicalize()
                .Normalize();*/
        cvy2=cvy;
        std::cout<<"\nm="<<m<<"\n";
        cvy.m=cvy2.m=m;
        cvy.decomposer=cvy2.decomposer=MatSVDFixedDim(m);
        cvy.Sweep();
        cvy.PrintSizes("cvy=");
        cvy2.Sweep();
        sb=Superblock( {&cvy,&mpo,&cvy} );
        sb_ya=Superblock( {&a,&cvy} );
        cvI[0]=cvy.CentralMat(nsite);
        cvI[1]=cvy2.CentralMat(nsite);
    }
    struct Target
    {
        TensorD w;
        double prob;
    };
    vector<Target> TargetsForDensityMatrix()
    {
        vector<Target> res;
        for(int i=0;i<n;i++)
        {
            res.push_back( {cvR[i],1.0/(2*n)} );
            res.push_back( {cvI[i],1.0/(2*n)} );
        }
        return res;
    }
    TensorD DensityMatrix(bool isLeft,int rpos)
    {
        vector<Target> target=TargetsForDensityMatrix();
        TensorD rho;
        for(Target& t:target)
        {
            auto x2= isLeft ? (t.w*cvy.right()).ReShape(rpos).Clone()
                            : (cvy.left()*t.w).ReShape(rpos).Clone().t() ;
            x2*=1.0/Norm(x2);
            rho+=x2*x2.t()*t.prob;
        }
        return rho;
    }
    void Solve()
    {
        auto Heff=sb.Oper(nsite);
        auto aeff=sb_ya.Oper(nsite)* a.CentralMat(nsite);
        for(int i=0;i<n;i++)
        {
            auto sol=FindCV(Heff,aeff,cvI[i],wR[i]+ener,wI[i],nIterMax);
            cvR[i]=sol.xR;
            cvI[i]=sol.xI;
            iter=sol.cIter;
        }
        cvy.setCentralMat(cvI[0]);
        cvy2.setCentralMat(cvI[1]);
        if (sb.pos.vx==1)
        {
            int mnext=cvy.right().dim.back();
            int rpos=2;
            if (cvy.pos.i==cvy.length-2)
            {
                mnext=cvy.right().dim.front();
                rpos=1;
            }
            if (nsite)
            {
                for (TensorD &t:cvR)
                    t=t.Decomposition(false,cvy.decomposer)[1];
                for (TensorD &t:cvI)
                    t=t.Decomposition(false,cvy.decomposer)[1];
            }
            auto rho=DensityMatrix(true,rpos);
            auto dcm=MatDensityFixedDimDecomp(rho.vec(),mnext);
            rot=TensorD({rho.dim[0],mnext},dcm.rot);
            cvy.decomposer=cvy2.decomposer=dcm;
        }
        else
        {
            int mnext=cvy.left().dim.front();
            int rpos=1;
            if (cvy.pos.i==0)
            {
                mnext=cvy.left().dim.back();
                rpos=2;
            }
            if (nsite)
            {
                for (TensorD &t:cvR)
                    t=t.Decomposition(true,cvy.decomposer)[0];
                for (TensorD &t:cvI)
                    t=t.Decomposition(true,cvy.decomposer)[0];
            }
            auto rho=DensityMatrix(false,rpos);
            auto dcm=MatDensityFixedDimDecomp(rho.vec(),mnext);
            rot=TensorD({rho.dim[0],mnext},dcm.rot);
            cvy.decomposer=cvy2.decomposer=dcm;
        }
        if (nsite)
        {
            sb.UpdateBlocks();
            sb_ya.UpdateBlocks();
        }
    }
    void SetPos(MPS::Pos p)
    {
        sb.SetPos(p);
        sb_ya.SetPos(p);
        cvy2.SetPos(p);
        cvI[0]=cvy.CentralMat(nsite);
        cvI[1]=cvy2.CentralMat(nsite);
    }
    std::vector<cmpx> Green(MPS& b,std::vector<cmpx> vz)
    {
        vector<cmpx> res;
        auto sb_yb=Superblock( {&b,&cvy} );
        for(int i=0;i<sb.length/2;i++)
        {
            SetPos({i,1});
            sb_yb.SetPos({i,1});
            Solve();
            Print();
        }
        auto cI0=cvy.CentralMat(nsite);
        auto Heff=sb.Oper(nsite);
        auto aeff=sb_ya.Oper(nsite)*a.CentralMat(nsite);
        auto beff=sb_yb.Oper(nsite)*b.CentralMat(nsite);
        for(cmpx z:vz)
        {
            double w=z.real(), eta=z.imag();
            auto cv=FindCV(Heff,aeff,cI0,w+ener,eta,2*nIterMax);
            std::cout<<cv.cIter<<" green VC iter; ";
            res.push_back( cmpx(Dot(beff,cv.xR),Dot(beff,cv.xI)) );
            cI0=cv.xI;
        }
        return res;
    }
    void Print() const
    {
        sb.Print();
        std::cout<<"; "<<iter<<" VC iter\n";
        std::cout.flush();
    }
};

#endif // DMRG_RES_CV_H
