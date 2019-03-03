#ifndef MPS_H
#define MPS_H

#include<iostream>
#include<vector>
#include<functional>
#include"tensor.h"



class MPS
{
private:
    std::vector<TensorD> M;
    int vx=1;
public:
    TensorD C;
    int length, m, pos=0;
    std::function< std::vector<TensorD>(const TensorD&,int) > t_decomposer;

    MPS(int length, int m)
        :M(length),length(length),m(m)
    {
        t_decomposer= [](const TensorD& t, int p)
        { return Decomposition(t,p); };
    }
    MPS(const std::vector<TensorD>& tensors,int m=0)
        :M(tensors),length(tensors.size()),m(m)
    {
        t_decomposer= [](const TensorD& t, int p)
                { return Decomposition(t,p); };

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
    MPS& FillRandu(Index dim)
    {
        FillNone(dim);
        for(TensorD &x:M) x.FillRandu();
        Canonicalize();
        return *this;
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
        auto usvt=t_decomposer(psi,psi.rank()-1);
        M[pos]=usvt.front();
        C=usvt[1];
        for(uint i=2;i<usvt.size();i++) C=C*usvt[i];
    }
    void SweepLeft()
    {
        if (pos==0) return;
        TensorD psi=ApplyC();
        if(vx==1) vx=-1; else pos--;
        ExtractNorm(psi);
        auto usvt=t_decomposer(psi,1);
        M[pos+1]=usvt.back();
        C=usvt[0];
        for(uint i=1;i<usvt.size()-1;i++) C=C*usvt[i];
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
inline MPO Fermi(int i, int L, bool dagged)
{
    static TensorD
            id( {1,2,2,1},{1,0,0,1} ),
            sg( {1,2,2,1}, {1,0,0,-1} );

    auto fe=dagged ? TensorD( {1,2,2,1}, {0,0,1,0} )
                   : TensorD( {1,2,2,1}, {0,1,0,0} );
    std::vector<TensorD> O(L);
    for(int j=0;j<L;j++)
    {
        O[j]= (j <i) ? sg :
              (j==i) ? fe :
                       id ;
    }
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
