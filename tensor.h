#ifndef TENSOR_H
#define TENSOR_H

#include<vector>
#include<random>
#include<iostream>
#include<fstream>
#include"utils.h"

using std::vector;

template<class T>
struct Tensor
{
    vector<int> dim;
    T* data=nullptr;

    Tensor() {}
    Tensor(const vector<int>& dim)
        :dim(dim),vd(Prod(dim))
    {
        data=vd.data();
    }


    void FillZeros()
    {
        std::fill(data,data+size(),T(0));
    }
    void FillRandu()
    {
        VecFillRandu(data,size() );
    }

    int size() const {  return Prod(dim);   }

    T& operator[](const vector<int>& id)
    {
        return data[Offset(id,dim)];
    }
    const T& operator[](const vector<int>& id) const
    {
        return data[Offset(id,dim)];
    }

    void Save(std::ostream& out) const
    {
        for(int x:dim) out<<x<<" ";
        out<<"\n";
        VecSave(data,size(),out);
    }
    void Save(std::string filename) const
    {
        std::ofstream out(filename); Save(out);
    }
    void Load(std::istream& in)
    {
        for(int x:dim) in>>x;
        VecLoad(data,size(),in);
    }
    void Load(std::string filename)
    {
        std::ifstream in(filename);  Load(in);
    }

    double Norm() const
    {
        return VecNorm(data,size());
    }
    friend double Norm(const Tensor& t)
    {
        return t.Norm();
    }

    void operator*=(double c)
    {
        VecProd(data,size(),c);
    }

    void operator-=(const Tensor& t2)
    {
        if (size()!=t2.size())
            throw std::invalid_argument("Tensor::operator-=");

        VecMinusInplace(data,t2.data,size());
    }
    Tensor operator-(const Tensor& t2) const
    {
        Tensor y=*this;
        y-=t2;
        return y;
    }

private:
    vector<T> vd;
};



#include"tensor.cpp"   // implementations

#endif // TENSOR_H
