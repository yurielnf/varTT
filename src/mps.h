#ifndef MPS_H
#define MPS_H

#include<iostream>
#include<vector>
#include"tensor.h"

class MPS
{
public:
    std::vector<TensorD> M;
    TensorD C;
    int length,m;
    double tol=1e-14;

    MPS(int length, int m)
        :M(length),length(length),m(m)
    {
        C=TensorD({1,1}, {1});
    }
    void FillNone(Index dim)
    {
        for(auto& x:M) x=TensorD(dim);
        Index dl=dim; dl.front()=1;
        Index dr=dim; dr.back()=1;
        M.front()=TensorD(dl);
        M.back ()=TensorD(dr);
    }
    void FillRandu(Index dim)
    {
        FillNone(dim);
        for(TensorD &x:M) x.FillRandu();
    }
    void PrintSizes() const
    {
        for(TensorD t:M)
        {
            for(int x:t.dim)
                std::cout<<" "<<x;
            std::cout<<",";
        }
        std::cout<<"\n";
    }
    void Normalize() {norm_n=1;}
    void Canonicalize()
    {
        if (pos!=-1) M[pos]=M[pos]*C;
        C=TensorD({1,1},{1});
        pos=-1;
        while(pos<(length+1)/2-1)
            SweepRight();
        auto cC=C;
        C=TensorD({1,1},{1});
        pos=length-1;
        while(pos>length/2-1)
            SweepLeft();
        C=cC*C;
        ExtractNorm(C);
    }
    void SetPos(int p)
    {
        while(pos<p) SweepRight();
        while(pos>p) SweepLeft();
    }
    void SweepRight()
    {
        if (pos>=length-1) return;
        pos++;
        auto psi=C*M[pos];
        ExtractNorm(psi);
        auto usvt=SVDecomposition(psi,psi.rank()-1);
        if (pos<length)
            M[pos]=usvt[0];
        C=usvt[1]*usvt[2];
    }
    void SweepLeft()
    {
        if (pos<0) return;
        auto psi=M[pos]*C;
        ExtractNorm(psi);
        auto usvt=SVDecomposition(psi,1);
        if (pos>=0)
            M[pos]=usvt[2];
        C=usvt[0]*usvt[1];
        pos--;
    }

    double norm() const {
        return pow(norm_n,length);
    }

    void operator*=(double c)
    {
        double nr=std::abs(c);
        norm_n*=pow(nr,1.0/length);
        C*=c/nr;
    }
    MPS operator*(double c) const { MPS A=*this; A*=c; return A; }
    void operator+=(const MPS& mps2)
    {
        MPS& mps1=*this;
        if (mps1.length != mps2.length)
            throw std::invalid_argument("MPO+MPO incompatible length");
        mps1.M.front() = DirectSum(mps1.M.front()*mps1.norm_n,
                                   mps2.M.front()*mps2.norm_n, true);
        for(int i=1;i<mps1.length-1;i++)
                mps1.M[i]=DirectSum(mps1.M[i]*mps1.norm_n,
                                    mps2.M[i]*mps2.norm_n);
        mps1.M.back() = DirectSum(mps1.M.back()*mps1.norm_n,
                                  mps2.M.back()*mps2.norm_n,false);

        mps1.C=DirectSum(mps1.C,mps2.C);
        mps1.norm_n=1;
        mps1.PrintSizes();
        mps1.Canonicalize();
        mps1.PrintSizes();
        //if ( mps1.NeedCompress() ) mps1.Compress();
    }

    bool NeedCompress() const
    {
        for(const TensorD& x:M)
            if (x.dim.front()>m || x.dim.back()>m)
                return true;
        return false;
    }
    void Compress()
    {
        Canonicalize();
        SetPos(length-2);
        SetPos(0);
        SetPos(length/2-1);
    }

private:
    void ExtractNorm(TensorD& psi)
    {
        double nr=Norm(psi);
        if (nr<tol)
            throw std::logic_error("mps:ExtractNorm() null matrix");
        norm_n*=pow(nr,1.0/length);
        psi*=1.0/nr;

        norm_n*=pow(Norm(C),1.0/length);
    }

    int pos=-1;
    double norm_n=1;                //norm(MPS)^(1/n)

};

struct MPSSum
{
    std::vector<MPS> v;
    int m;

    MPSSum(int m)
        :m(m){}

    void operator+=(const MPS& mps)
    {
        v.push_back(mps);
        v.back().m=m;
    }
    void operator+=(const MPSSum& s) { for(auto x:s.v) (*this)+=x; }
    operator MPS() { return VecReduce(v.data(),v.size()); }
};

typedef MPS MPO;

//---------------------------- Helpers ---------------------------

inline MPO MPOIdentity(int length, int d)
{
    MPS Id(length,1);
    Id.FillNone({1,d,d,1});
    for(auto& x:Id.M) x.FillEye(2);
    Id.Canonicalize();
    return Id;
}

#endif // MPS_H
