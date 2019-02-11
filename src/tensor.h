#ifndef TENSOR_H
#define TENSOR_H

#include<vector>
#include<random>
#include<iostream>
#include<fstream>
#include<array>
#include<algorithm>
#include<functional>
#include<type_traits>

#include"utils.h"
#include"index.h"

template<class Container=std::vector<double>> struct Tensor;

//template<class T> using TensorReShape=Tensor<T, std::vector<T>& >;
//template<class T> using TensorReShapeC=Tensor<T, const std::vector<T>& >;
template<class C> struct TensorNotation;

using TensorD=Tensor<std::vector<double>>;


template<typename C>
struct Tensor
{
    Index  dim;
    C v;
    std::string id;

    typedef typename std::decay<C>::type _C;
    typedef typename _C::value_type T;

    Tensor() {}
    Tensor(const Index& dim)
        :dim(dim),v(Prod(dim)) {}
    Tensor(const Index& dim,const  C& v)
        :dim(dim),v(v) {}


    template<class C2>
    Tensor& operator =(const Tensor<C2>& t)
    {
        auto tmp=t.Reorder(t.id,id);
        dim=tmp.dim;
        v=tmp.v;
        id=tmp.id;
        return *this;
    }

    int rank() const {return dim.size();}

    void FillZeros()
    {
        for(T& x:v) x=0;
    }
    void FillRandu()
    {
        VecFillRandu(v.data(),v.size());
    }

    T& operator[](const Index& id)
    {
        return v[Offset(id,dim)];
    }
    const T& operator[](const Index& id) const
    {
        return v[Offset(id,dim)];
    }

    Tensor& operator()(std::string id)
    {
        this->id=id;
        return *this;
//        return {*this,id};
    }

    void Save(std::ostream& out) const
    {
        for(int x:dim) out<<x<<" ";
        out<<"\n";
        VecSave(v.data(),v.size(),out);
    }
    void Save(std::string filename) const
    {
        std::ofstream out(filename); Save(out);
    }
    void Load(std::istream& in)
    {
        for(int x:dim) in>>x;
        VecLoad(v.data(),v.size(),in);
    }
    void Load(std::string filename)
    {
        std::ifstream in(filename);  Load(in);
    }

    void operator*=(T c)
    {
        VecProd(v.data(),v.size(),c);
    }
    template<class C2>
    void operator+=(const Tensor<C2>& t2)
    {
        if (dim!=t2.dim)
            throw std::invalid_argument("Tensor::operator-= incompatible");

        VecPlusInplace(v.data(),t2.v.data(),v.size());
    }
    template<class C2>
    void operator-=(const Tensor<C2>& t2)
    {
        if (dim!=t2.dim)
            throw std::invalid_argument("Tensor::operator-=incompatible");

        VecMinusInplace(v.data(),t2.v.data(),v.size());
    }

    Tensor<_C> operator-() const
    {
        Tensor y=*this;
        VecNegativeInplace(y.v.data(),v.size());
        return y;
    }
    Tensor<_C> operator*(T c) const
    {
        Tensor<_C> y=*this;
        y*=c;
        return y;
    }
    Tensor<_C> operator+(const Tensor& t2) const
    {
        Tensor<_C> y=*this;
        y+=t2;
        return y;
    }
    Tensor<_C> operator-(const Tensor& t2) const
    {
        Tensor<_C> y=*this;
        y-=t2;
        return y;
    }

    Tensor<_C&> ReShape(int splitPos)
    {
        auto dim_v=SplitIndex(dim,splitPos);
        return {{ Prod(dim_v[0]), Prod(dim_v[1])}, v};
    }
    Tensor<const _C&> ReShape(int splitPos) const
    {
        auto dim_v=SplitIndex(dim,splitPos);
        return {{ Prod(dim_v[0]), Prod(dim_v[1])}, v};
    }
//    Tensor<C&> ReShape(std::vector<int> splitPos)
//    {
//        auto dim_v=SplitIndex(dim,splitPos);
//        Index dimr(dim_v.size());
//        for(int i=0;i<dim_v.size();i++)
//            dimr[i]=Prod(dim_v[i]);
//        return TensorReShape<T>(dimr,v);
//    }
//    Tensor<const C&> ReShape(std::vector<int> splitPos) const
//    {
//        auto dim_v=SplitIndex(dim,splitPos);
//        Index dimr(dim_v.size());
//        for(int i=0;i<dim_v.size();i++)
//            dimr[i]=Prod(dim_v[i]);
//        return TensorReShapeC<T>(dimr,v);
//    }

    Tensor<_C> DiagMat() const
    {
        int n=v.size();
        Tensor<_C> t2({n,n});
        t2.FillZeros();
        for(int i=0;i<n;i++)
            t2[{i,i}]=v[i];
        return t2;
    }

    Tensor<_C> Reorder(std::vector<int> posMap) const
    {
        Tensor<_C> t(IndexReorder(dim,posMap));
        for(uint i=0;i<v.size();i++)
        {
            int im=Offset(IndexReorder(ToIndex(i,dim),posMap),t.dim) ;
            t.v[im]=v[i];
        }
        return t;
    }
    Tensor<_C> Reorder(std::string ini,std::string fin) const
    {
        if (ini==fin) return *this;
        return Reorder(Permutation(ini,fin));
    }
    Tensor<_C> Transpose(int splitPos) const
    {
        auto dim_v=SplitIndex(dim,splitPos);
        std::swap(dim_v[0],dim_v[1]);
        Index dim2;
        for(Index d:dim_v)
            for(auto x:d)
                dim2.push_back(x);
        Tensor<_C> t2(dim2);
        auto m1=ReShape(splitPos);
        MatTranspose(m1.v.data(),t2.v.data(),m1.dim[0],m1.dim[1]);
        return t2;
    }

    template<class C2>
    Tensor<_C> operator*(const Tensor<C2>& t2) const
    {
        const Tensor &t1=*this;
        Tensor t3(IndexMul(t1.dim,t2.dim));
        Multiply(t1,t2,t3);
        return t3;
    }


};


//-------------------------------------- friend functions -----------------------------------

template<class C1,class C2>
bool operator==(const Tensor<C1>& t1,const Tensor<C2>& t2)
{
    return (t1.dim==t2.dim && t1.v==t2.v);
}

template<class C1,class C2>
bool operator!=(const Tensor<C1>& t1,const Tensor<C2>& t2)
{
    return !(t1==t2);
}

template<class C>
std::ostream& operator<<(std::ostream& out,const Tensor<C>& M)
{
    M.Save(out); return out;
}
template<class C>
std::istream& operator>>(std::istream& in,Tensor<C>& t)
{
    t.Load(in); return in;
}
template<class C1, class C2>
T Dot(const Tensor<C1>& t1,const Tensor<C2>& t2)
{
    return VecDot(t1.data(),t2.data(),t1.size());
}
template<class C>
double Norm(const Tensor<C>& t)
{
    return VecNorm(t.v.data(),t.v.size());
}
template<class C1,class C2,class C3>
void Multiply(const Tensor<C1>& t1,const Tensor<C2>& t2, Tensor<C3>& t3)
{
    Index dim_r=IndexMul(t1.dim,t2.dim);
    if (dim_r!=t3.dim)
        throw std::invalid_argument("Tensor:: Multiply() incompatible dimensions");

    auto m1=t1.ReShape(t1.rank()-1);  //Matrix operation
    auto m2=t2.ReShape(1);
    auto m3=t3.ReShape(t1.rank()-1);
    MatMul(m1.v.data(),m2.v.data(),m3.v.data(),m1.dim[0],m1.dim[1],m2.dim[1]);
}
template<class C>
std::array<Tensor<C>,2> EigenDecomposition(const Tensor<C>& t,int splitPos)
{
    auto mt=t.ReShape(splitPos);
    if (mt.dim[0]!=mt.dim[1])
        throw std::invalid_argument("Tensor:: EigenDecomposition non-square matrix");
    int n=mt.dim[0];
    auto dimEvec=SplitIndex(t.dim,splitPos)[0];
    dimEvec.push_back(n);    //evec dimension
    auto evec=Tensor(dimEvec);
    auto eval=Tensor({n});
    MatFullDiag(mt.data(),n,evec.data(),eval.data());
    return {evec,eval};
}
friend std::vector<Tensor> SVDecomposition(const Tensor& t,int splitPos) //M=U*S*Vt
{
    auto mt=t.ReShape(splitPos);
    int n=std::min(mt.dim[0],mt.dim[1]);
    auto dimUV=SplitIndex(t.dim,splitPos);
    dimUV[0].push_back(n);    //U dimension
    dimUV[1].push_back(n);    //V dimension
    auto U=Tensor(dimUV[0]); U.FillZeros();
    auto S=Tensor({n});      S.FillZeros();
    auto V=Tensor(dimUV[1]); V.FillZeros();
    MatSVD(mt.v.data(),mt.dim[0],mt.dim[1],U.v.data(),S.v.data(),V.v.data());
    return {U,S.DiagMat(),V.Transpose(V.rank()-1)};
}



template<class Tensor>
struct TensorNotation
{
    Tensor t;
    std::string id;
    TensorNotation(const Tensor& t, std::string id)
        :t(t),id(id) {}
    TensorNotation(const TensorNotation<typename std::remove_reference<Tensor>::type>& tn)
        :t(tn.t),id(tn.id) {}
    TensorNotation(TensorNotation<typename std::remove_reference<Tensor>::type>&& tn)
        :t(tn.t),id(tn.id) {}

    //auto Reorder(std::string fin) const { return t.Reorder(id,fin); }
    TensorNotation& operator=(const TensorNotation& tn)
    {
        t=tn.t.Reorder(tn.id,id);
        return *this;
    }

    TensorNotation<typename std::remove_reference<Tensor>::type> operator*(const TensorNotation& tn)
    {
        auto ids=SortForMultiply(id,tn.id);
        auto t1=   t.Reorder(id   ,ids[0]);
        auto t2=tn.t.Reorder(tn.id,ids[1]);
        auto t3=t1*t2;
        auto tn3=TensorNotation<typename std::remove_reference<Tensor>::type>(t3,ids[2]);
        return tn3;

    }

};

#include"tensor.cpp"   // implementations

#endif // TENSOR_H
