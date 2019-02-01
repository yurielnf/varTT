#ifndef TENSOR_H
#define TENSOR_H

#include<vector>
#include<random>
#include<iostream>
#include<fstream>
#include<array>

#include"utils.h"
#include"index.h"

template<class T>
struct Tensor
{
    const Index  dim;
    const int size=0;
private:
    std::vector<T> vd;
public:
     T* const data=nullptr;

    Tensor() {}
    Tensor(const Index& dim)
        :dim(dim),size(Prod(dim)),vd(size),data(vd.data()) {}
    Tensor(const T* data, const Index& dim)
        :dim(dim),size(Prod(dim)),data(data) {}

    void FillZeros()
    {
        std::fill(data,data+size,T(0));
    }
    void FillRandu()
    {
        VecFillRandu(data,size );
    }


    int rank() const { return dim.size();}

    T& operator[](const Index& id)
    {
        return data[Offset(id,dim)];
    }
    const T& operator[](const Index& id) const
    {
        return data[Offset(id,dim)];
    }

    void Save(std::ostream& out) const
    {
        for(int x:dim) out<<x<<" ";
        out<<"\n";
        VecSave(data,size,out);
    }
    void Save(std::string filename) const
    {
        std::ofstream out(filename); Save(out);
    }
    void Load(std::istream& in)
    {
        for(int x:dim) in>>x;
        VecLoad(data,size,in);
    }
    void Load(std::string filename)
    {
        std::ifstream in(filename);  Load(in);
    }

    void operator*=(T c)
    {
        VecProd(data,size(),c);
    }
    void operator+=(const Tensor& t2)
    {
        if (size()!=t2.size())
            throw std::invalid_argument("Tensor::operator-=");

        VecPlusInplace(data,t2.data,size());
    }
    void operator-=(const Tensor& t2)
    {
        if (dim!=t2.dim)
            throw std::invalid_argument("Tensor::operator-=");

        VecMinusInplace(data,t2.data,size);
    }

    Tensor operator-() const
    {
        Tensor y=*this;
        VecNegativeInplace(y.data,size());
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


    Tensor ReShape(int splitPos) const
    {
        auto dim_v=SplitIndex(dim,splitPos);
        return Tensor(data, { Prod(dim_v[0]), Prod(dim_v[1])} );
    }
    Tensor ReShape(std::vector<int> splitPos) const
    {
        auto dim_v=SplitIndex(dim,splitPos);
        Index dimr(dim_v.size());
        for(int i=0;i<dim_v.size();i++)
            dimr[i]=Prod(dim_v[i]);
        return Tensor(data, dimr );
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
        return VecDot(t1.data,t2.data,t1.size());
    }
    friend double Norm(const Tensor& t)
    {
        return VecNorm(t.data,t.size);
    }
    friend void Multiply(const Tensor& t1,const Tensor& t2, Tensor& t3)
    {
        Index dim_r=IndexMul(t1.dim,t2.dim);
        if (dim_r!=t3.dim)
            throw std::invalid_argument("Tensor:: Multiply() incompatible dimensions");

        Tensor m1=t1.ReShape(t1.dim.size()-1);  //Matrix operation
        Tensor m2=t2.ReShape(1);
        Tensor m3=t3.ReShape(t1.dim.size()-1);
        MatMul(m1.data,m2.data,m3.data,m1.dim[0],m1.dim[1],m2.dim[1]);
    }

    friend std::array<Tensor,2> EigenDecomposition(const Tensor& t,int splitPos)
    {
        auto mt=t.ReShape(splitPos);
        if (mt.dim[0]!=mt.dim[1])
            throw std::invalid_argument("Tensor:: EigenDecomposition non-square matrix");
        auto dim_v=SplitIndex(t.dim,splitPos);
        dim_v[0].push_back(mt.dim[0]);    //evec dimension
        std::array<Tensor,2> res;
        res[0]=Tensor(dim_v[0]);          //evec
        res[1]=Tensor({mt.dim[0]});       //eval
        MatFullDiag(mt.data,mt.dim[0],res[0].data,res[1].data);
        return res;
    }
};




#include"tensor.cpp"   // implementations

#endif // TENSOR_H
