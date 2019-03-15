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
    int vx=-1;
public:
    TensorD C;
    int length, m, pos=0;
    enum class decomp_type {svd,eye};
    decomp_type t_decomposer;

    MPS(int length, int m, decomp_type decomp=decomp_type::svd)
        :M(length),length(length),m(m),t_decomposer(decomp)
    {
    }
    MPS(const std::vector<TensorD>& tensors,int m=0, decomp_type decomp=decomp_type::svd)
        :M(tensors),length(tensors.size()),m(m),t_decomposer(decomp)
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
        static TensorD one({1},{1});
        TensorD tr=one*pow(norm_n,length);
        for(int i=0;i<length;i++)
        {
            tr=tr*M[i];
            if (i==pos) tr=tr*C;
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
        for(int i=0;i<2;i++) Sweep();
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
        std::array<TensorD,2> uv;
        if (t_decomposer==decomp_type::svd)
            uv=psi.Decomposition(false,MatSVD);
        else
            uv=psi.Decomposition(false,MatChopDecomp);
        M[pos]=uv[0];
        C=uv[1];
    }
    void SweepLeft()
    {
        if (pos==0) return;
        TensorD psi=ApplyC();
        if(vx==1) vx=-1; else pos--;
        ExtractNorm(psi);
        std::array<TensorD,2> uv;
        if (t_decomposer==decomp_type::svd)
            uv=psi.Decomposition(true,MatSVD);
        else
            uv=psi.Decomposition(true,MatChopDecomp);
        M[pos+1]=uv[1];
        C=uv[0];
    }
    void SetPos(int p)
    {
        if (pos<0 || pos>length-2)
            throw std::invalid_argument("SB::SetPos out of range");
        while(pos<p) SweepRight();
        while(pos>p) SweepLeft();
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
        if ( mps1.MaxVirtDim()>m )
            for(int i=0;i<2;i++)
                mps1.Sweep();
    }
    MPS operator*(const MPS& mps2) const
    {
        const MPS& mps1=*this;
        if (mps1.length != mps2.length)
            throw std::invalid_argument("MPO+MPO incompatible length");
        MPS mps3(mps1.length,mps1.m*mps2.m);
        for(int i=0;i<length;i++)
        {
            TensorD tr;
            tr("iIjJlL")=mps1.M[i]("ijkl")*mps2.M[i]("IkJL");
            mps3.M[i]=tr.ReShape({2,3,4}).Clone();
        }
        TensorD tr;
        tr("iIjJ")=mps1.C("ij")*mps2.C("IJ");
        mps3.C=tr.ReShape(2).Clone();
        mps3.norm_n = mps1.norm_n * mps2.norm_n;
//        if ( mps3.MaxVirtDim()>m ) mps3.Sweep();
        return mps3;
    }

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
    MPS::decomp_type t_decomposer;

    MPSSum(int m,MPS::decomp_type t_decomposer=MPS::decomp_type::svd)
        :m(m)
        ,t_decomposer(t_decomposer)
    {}

    void operator+=(const MPS& mps)
    {
        v.push_back(mps);
        v.back().m=m;
        v.back().t_decomposer=t_decomposer;
    }
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
            id( {1,2,2,1}, {1,0,0,1} ),
            sg( {1,2,2,1}, {1,0,0,-1} ),
            cd( {1,2,2,1}, {0,1,0,0} ),
            c ( {1,2,2,1}, {0,0,1,0} );

    auto fe=dagged ? cd : c;
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

inline MPO HamTB2(int L,bool periodic)
{
    const int m=128;
    MPSSum h(m);
    for(int i=0;i<L-1+periodic; i++)
    {
        h += Fermi(i,L,true)*Fermi((i+1)%L,L,false)*(-1.0) ;
        h += Fermi((i+1)%L,L,true)*Fermi(i,L,false)*(-1.0) ;
    }
    return h.toMPS();
}

inline MPO HamTBExact(int L)
{
    static TensorD
            I ={{2,2},{1,0,0,1}},
            c ={{2,2},{0,0,1,0}},
            sg={{2,2},{1,0,0,-1}},
            o ={{2,2},{0,0,0,0}};
    TensorD cd=c.t(),n=cd*c, H=o;
    auto fv1=flat({I, H , c*sg*(-1.0), cd*sg, o});
    auto fv2=flat(   { I, H , c*sg*(-1.0), cd*sg, o,      //H
                       o, I , o    , o          , o,      //I
                       o, cd, o    , o          , o,      //cd
                       o, c , o    , o          , o,      //c
                       o, n , o    , o          , I,      //nT
                     }  );
    auto fv3=flat({H,I,cd,c,n});

    std::vector<TensorD> O(L);
    O[0]={ {2,2,5,1}, fv1 };
    for(int i=1;i<L-1;i++)
        O[i]={ {2,2,5,5}, fv2 };
    O[L-1]=TensorD( {2,2,1,5},fv3 );
    for(TensorD& x:O) x=x.Reorder("ijkl","lijk");
    return MPS(O,5,MPS::decomp_type::eye)*(-1.0);
}

#endif // MPS_H
