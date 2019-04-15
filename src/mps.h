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
public:
    TensorD C;
    int length, m;
//    typedef std::array<int,2> Pos;

    struct Pos
    {
        int i,vx;
        Pos& operator++()
        {
            if (vx==-1) vx=1;
            else ++i;
            return *this;
        }
        Pos& operator--()
        {
            if (vx==1) vx=-1;
            else --i;
            return *this;
        }

        bool operator<(Pos p) const { return id()<p.id(); }
        bool operator>(Pos p) const { return p<(*this); }
        bool operator==(Pos p) const { return id()==p.id(); }
    private:
        int id() const {return i*2+vx;}
    };

    Pos pos=Pos({0,-1});
    TensorD::mat_decomp decomposer;

    MPS():length(0),m(0){}
    MPS(int length, int m,
        TensorD::mat_decomp decomposer=MatQRDecomp)
        :M(length),length(length),m(m),decomposer(decomposer)
    {}
    MPS(const std::vector<TensorD>& tensors,int m=0,
        TensorD::mat_decomp decomposer=MatQRDecomp)
        :M(tensors),length(tensors.size()),m(m),decomposer(decomposer)
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
    MPS& FillRandu(Index dim)
    {
        FillNone(dim);
        for(TensorD &x:M) x.FillRandu();
//        Canonicalize();
        return *this;
    }
    const TensorD& at(int i) const { return M[i]; }
    void PrintSizes(const char str[]="") const
    {
        std::cout<<str<<"\n";
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
        static const TensorD one({1},{1});
        TensorD tr=one*pow(norm_n,length);
        for(int i=0;i<length;i++)
        {
            tr=tr*M[i];
            if (i==pos.i) tr=tr*C;
        }
        return tr*one;
    }
    MPS& Normalize() {norm_n=1; C*=1.0/Norm(C); return *this;}
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
        auto decom=decomposer;
        decomposer=MatQRDecomp;
        Sweep();
        decomposer=decom;
        return *this;
    }
    MPS& Sweep()
    {
        for(auto pi:SweepPosSec(length))
            SetPos(pi);
        return *this;
    }
    TensorD ApplyC()
    {
        return pos.vx==1 ? C*M[pos.i+1] : M[pos.i]*C;
    }
//    void SweepG()
//    {
//        TensorD psi=ApplyC();
//        ExtractNorm(psi);
//        pos=pos+vx;
//        if (pos+vx>length-1 || pos+vx<0) vx=-vx;
//        bool vxb= vx==1?false:true;
//        auto uv=psi.Decomposition(vxb,decomposer);
//        M[pos]=uv[0+vxb];
//        C=uv[1-vxb];
//    }

    void SweepRight()
    {
        if ( pos.i==length-2 && pos.vx==1 ) return;
        TensorD psi=ApplyC();
        ++pos;
        ExtractNorm(psi);
        std::array<TensorD,2> uv;
        uv=psi.Decomposition(false,decomposer);
        M[pos.i]=uv[0];
        C=uv[1];
    }
    void SweepLeft()
    {
        if ( pos.i==0 && pos.vx==-1 ) return;
        TensorD psi=ApplyC();
        --pos;
        ExtractNorm(psi);
        std::array<TensorD,2> uv;
        uv=psi.Decomposition(true,decomposer);
        M[pos.i+1]=uv[1];
        C=uv[0];
    }
    void SetPos(Pos p)
    {
        if (p.i<0 || p.i>length-2)
            throw std::invalid_argument("SB::SetPos out of range");
        while(pos<p) SweepRight();
        while(pos>p) SweepLeft();
//        if (pos.vx==-1 && p.vx==1) SweepRight();
//        if (pos.vx==1 && p.vx==-1) SweepLeft();
    }
    double norm_factor() const
    {
        return pow(norm_n,length);
    }
    double norm() const
    {
        return pow(norm_n,length)*Norm(C);
    }

    void operator*=(double c)
    {
        double nr=std::fabs(c);
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
        mps1.Canonicalize();
        if ( mps1.MaxVirtDim()>m )
            for(int i=0;i<1;i++)
                mps1.Sweep();
    }
    void operator*=(const MPS& mps2)

    {
        MPS& mps1=*this;
        if (mps1.length != mps2.length)
            throw std::invalid_argument("MPS*MPS incompatible length");
//        MPS mps3(mps1.length,mps1.m*mps2.m);
        for(int i=0;i<length;i++)
        {
            TensorD tr;
            if (mps2.M[i].rank()==3)
            {
                tr("iIjlL")=mps1.M[i]("ijkl")*mps2.M[i]("IkL");
                mps1.M[i]=tr.ReShape({2,3}).Clone();
            }
            else if (mps2.M[i].rank()==4)
            {
                tr("iIjJlL")=mps1.M[i]("ijkl")*mps2.M[i]("IkJL");
                mps1.M[i]=tr.ReShape({2,3,4}).Clone();
            }
        }
        TensorD tr;
        tr("iIjJ")=mps1.C("ij")*mps2.C("IJ");
        mps1.C=tr.ReShape(2).Clone();
        mps1.norm_n = mps1.norm_n * mps2.norm_n;
        if ( mps1.MaxVirtDim()>m ) mps1.Sweep();
    }
    MPS operator*(const MPS& mps2) const
    {   MPS res=*this; res*=mps2; return res; }

    int MaxVirtDim() const
    {
        int mm=std::numeric_limits<int>::min();
        for(const TensorD& x:M)
        {
            mm=std::max(mm,x.dim.front());
            mm=std::max(mm,x.dim.back());
        }
        return mm;
    }    
    static std::vector<Pos> SweepPosSec(int length)
    {
        std::vector<Pos> pos;
        for(int i=length/2-1;i<length-1;i++) //Right
            pos.push_back({i,1});
        for(int i=length-2;i>=0;i--)
            pos.push_back({i,-1});
        for(int i=0;i<length/2;i++)
            pos.push_back({i,1});
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
    TensorD::mat_decomp decomposer;

    MPSSum(int m,TensorD::mat_decomp decomposer=MatQRDecomp)
        :m(m)
        ,decomposer(decomposer)
    {}

    void operator+=(const MPS& mps)
    {
        v.push_back(mps);
        v.back().m=m;
        v.back().decomposer=decomposer;
    }
    MPS toMPS() const
    {
        auto vc=v;
        return VecReduce(vc.data(),vc.size()).Canonicalize().Sweep();
    }
};

typedef MPS MPO;

//---------------------------- Helpers ---------------------------

 MPO MPOIdentity(int length);
 MPO Fermi(int i, int L, bool dagged);
 MPO MPOEH(int length);

 MPO HamTbAuto(int L,bool periodic);
 MPO HamTBExact(int L);

#endif // MPS_H
