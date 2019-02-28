#ifndef MPS_H
#define MPS_H

#include<iostream>
#include<vector>
#include"tensor.h"



class MPS
{
private:
    std::vector<TensorD> M;
    int vx=1;
public:
    TensorD C;
    int length, m, pos=0;

    MPS(int length, int m)
        :M(length),length(length),m(m)
    {}
    MPS(const std::vector<TensorD>& tensors,int m=0)
        :M(tensors),length(tensors.size()),m(m)
    {
        int n0=M[0].dim.back();
        C=TensorD({n0,n0});
        C.FillEye(1);
        Canonicalize();
    }
    void FillNone(Index dim)
    {
        for(auto& x:M) x=TensorD(dim);
        Index dl=dim; dl.front()=1;
        Index dr=dim; dr.back()=1;
        M.front()=TensorD(dl);
        M.back ()=TensorD(dr);

        int n0=M[0].dim.back();
        C=TensorD({n0,n0});
        C.FillEye(1);
    }
    void FillRandu(Index dim)
    {
        FillNone(dim);
        for(TensorD &x:M) x.FillRandu();
        Canonicalize();
    }
    const TensorD& at(int i) const { return M[i]; }
    void PrintSizes() const
    {
        for(const TensorD& t:M)
        {
            for(int x:t.dim)
                std::cout<<" "<<x;
            std::cout<<",";
        }
        std::cout<<"\n";
    }
    operator TensorD() const
    {
        TensorD tr=TensorD({1,1},{1})*pow(norm_n,length);
        if (pos==-1) tr=C*tr;
        for(int i=0;i<length;i++)
        {
            tr=tr*M[i];
            if (i==pos) tr=tr*C;
        }
        return tr;
    }
    void Normalize() {norm_n=1;C*=1.0/Norm(C);}
    MPS& Canonicalize()
    {
//        SetPos(0);
//        SetPos(length/2-1);
//        auto cC=C;
//        C=TensorD({1,1},{1});
//        pos=length-2;
//        SetPos(length/2-1);
//        C=cC*C;
//        ExtractNorm(C);
        Sweep();
        return *this;
    }
    void Sweep() { for(int i:SweepPosSec(length)) SetPos(i); }

    TensorD ApplyC()
    {
        return vx==1 ? C*M[pos+1] : M[pos]*C;
    }

    void SweepRight()
    {
        if (pos==length-2) return;
        TensorD psi=ApplyC();
        if (vx==-1) vx=1; else pos++;
        ExtractNorm(psi);
        auto usvt=SVDecomposition(psi,psi.rank()-1);
        M[pos]=usvt[0];
        C=usvt[1]*usvt[2];
    }
    void SweepLeft()
    {
        if (pos==0) return;
        TensorD psi=ApplyC();
        if(vx==1) vx=-1; else pos--;
        ExtractNorm(psi);
        auto usvt=SVDecomposition(psi,1);
        M[pos+1]=usvt[2];
        C=usvt[0]*usvt[1];
    }
    void SetPos(int p)
    {
        if (pos<0 || pos>length-2)
            throw std::invalid_argument("SB::SetPos out of range");
        while(pos<p) SweepRight();
        while(pos>p) SweepLeft();
    }
    double norm() const
    {
        return pow(norm_n,length)*Norm(C);
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
        if ( mps1.NeedCompress() ) mps1.Sweep();
    }

    bool NeedCompress() const
    {
        for(const TensorD& x:M)
            if (x.dim.front()>m || x.dim.back()>m)
                return true;
        return false;
    }
    static std::vector<int> SweepPosSec(int length)
    {
        std::vector<int> pos;
        for(int i=length/2-1;i<length-1;i++) //Right
            pos.push_back(i);
        for(int i=length-2;i>=0;i--)
            pos.push_back(i);
        for(int i=0;i<length/2;i++)
            pos.push_back(i);
        return pos;
    }

private:
    void ExtractNorm(TensorD& psi)
    {
        double nr=Norm(psi);
        if (nr==0)
            throw std::logic_error("mps:ExtractNorm() null matrix");
        norm_n*=pow(nr,1.0/length);
        psi*=1.0/nr;
    }

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
//    void operator+=(const MPSSum& s) { for(auto x:s.v) (*this)+=x; }
    MPS toMPS() const
    {
        auto vc=v;
        return VecReduce(vc.data(),vc.size()).Canonicalize();
    }
};

typedef MPS MPO;

//---------------------------- Helpers ---------------------------

inline MPO MPOIdentity(int length)
{
    std::vector<TensorD> O(length);
    std::vector<double> id={1,0,0,1};
    for(auto& x:O) x=TensorD({1,2,2,1}, id);
    return O;
}
inline MPO MPOEH(int length)
{
    std::vector<TensorD> O(length);
    std::vector<double> eh={1.3,-2,-1,0}, id={1,0,0,1};
    for(uint i=0;i<O.size();i++)
        if(i%2==0)
            O[i]=TensorD({1,2,2,1}, eh);
        else
            O[i]=TensorD({1,2,2,1}, id);
    return O;
}

#endif // MPS_H
