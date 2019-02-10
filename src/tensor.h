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
    typedef typename std::decay<C>::type::value_type T;

    Tensor() {}
    Tensor(const Index& dim)
        :dim(dim),v(Prod(dim)) {}
    Tensor(const Index& dim,const  C& v)
        :dim(dim),v(v) {}

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

    TensorNotation<Tensor&> operator()(std::string id)
    {
        return {*this,id};
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
    void operator+=(const Tensor& t2)
    {
        if (dim!=t2.dim)
            throw std::invalid_argument("Tensor::operator-= incompatible");

        VecPlusInplace(v.data(),t2.v.data(),v.size());
    }
    void operator-=(const Tensor& t2)
    {
        if (dim!=t2.dim)
            throw std::invalid_argument("Tensor::operator-=incompatible");

        VecMinusInplace(v.data(),t2.v.data(),v.size());
    }

    Tensor operator-() const
    {
        Tensor y=*this;
        VecNegativeInplace(y.v.data(),v.size());
        return y;
    }
    Tensor operator*(T c) const
    {
        Tensor y=*this;
        y*=c;
        return y;
    }
    Tensor operator+(const Tensor& t2) const
    {
        Tensor y=*this;
        y+=t2;
        return y;
    }
    Tensor operator-(const Tensor& t2) const
    {
        Tensor y=*this;
        y-=t2;
        return y;
    }

    Tensor<C&> ReShape(int splitPos)
    {
        auto dim_v=SplitIndex(dim,splitPos);
        return {{ Prod(dim_v[0]), Prod(dim_v[1])}, v};
    }
    Tensor<const C&> ReShape(int splitPos) const
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

    Tensor DiagMat() const
    {
        int n=v.size();
        Tensor t2({n,n});
        t2.FillZeros();
        for(int i=0;i<n;i++)
            t2[{i,i}]=v[i];
        return t2;
    }

    Tensor Reorder(std::vector<int> posMap) const
    {
        Tensor t(IndexReorder(dim,posMap));
        for(uint i=0;i<v.size();i++)
        {
            int im=Offset(IndexReorder(ToIndex(i,dim),posMap),t.dim) ;
            t.v[im]=v[i];
        }
        return t;
    }
    Tensor Reorder(std::string ini,std::string fin) const
    {
        return Reorder(Permutation(ini,fin));
    }
    Tensor Transpose(int splitPos) const
    {
        auto dim_v=SplitIndex(dim,splitPos);
        std::swap(dim_v[0],dim_v[1]);
        Index dim2;
        for(Index d:dim_v)
            for(auto x:d)
                dim2.push_back(x);
        Tensor t2(dim2);
        auto m1=ReShape(splitPos);
        MatTranspose(m1.v.data(),t2.v.data(),m1.dim[0],m1.dim[1]);
        return t2;
    }

    Tensor operator*(const Tensor& t2) const
    {
        const Tensor &t1=*this;
        Tensor t3(IndexMul(t1.dim,t2.dim));
        Multiply(t1,t2,t3);
        return t3;
    }

    friend bool operator==(const Tensor& t1,const Tensor& t2)
    {
        return (t1.dim==t2.dim && t1.v==t2.v);
    }
    friend bool operator!=(const Tensor& t1,const Tensor& t2)
    {
        return (t1.dim!=t2.dim || t1.v!=t2.v);
    }

    friend std::ostream& operator<<(std::ostream& out,const Tensor& M)
    {
        M.Save(out); return out;
    }
    friend std::istream& operator>>(std::istream& in,Tensor& t)
    {
        t.Load(in); return in;
    }
    friend T Dot(const Tensor& t1,const Tensor& t2)
    {
        return VecDot(t1.data(),t2.data(),t1.size());
    }
    friend double Norm(const Tensor& t)
    {
        return VecNorm(t.v.data(),t.v.size());
    }
    friend void Multiply(const Tensor& t1,const Tensor& t2, Tensor& t3)
    {
        Index dim_r=IndexMul(t1.dim,t2.dim);
        if (dim_r!=t3.dim)
            throw std::invalid_argument("Tensor:: Multiply() incompatible dimensions");

        auto m1=t1.ReShape(t1.rank()-1);  //Matrix operation
        auto m2=t2.ReShape(1);
        auto m3=t3.ReShape(t1.rank()-1);
        MatMul(m1.v.data(),m2.v.data(),m3.v.data(),m1.dim[0],m1.dim[1],m2.dim[1]);
    }

    friend std::array<Tensor,2> EigenDecomposition(const Tensor& t,int splitPos)
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

};

template<class Tensor>
struct TensorNotation
{
    Tensor t;
    std::string id;


    //auto Reorder(std::string fin) const { return t.Reorder(id,fin); }
    TensorNotation& operator=(const TensorNotation& tn)
    {
        t=tn.t.Reorder(tn.id,id);
        return *this;
    }

    TensorNotation operator*(const TensorNotation& tn)
    {
        auto ids=SortForMultiply(id,t.id);
        auto t1=   t.Reorder(id   ,ids[0]);
        auto t2=tn.t.Reorder(tn.id,ids[1]);
        auto t3=t1*t2;
        return TensorNotation<std::remove_reference_t<Tensor>> tn3(t3,IndexMatMul);

    }

};

#include"tensor.cpp"   // implementations

#endif // TENSOR_H
