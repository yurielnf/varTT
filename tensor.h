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
    vector<int> const dim;
private:
    vector<T> vd;
public:
     T* const data=nullptr;

    Tensor() {}
    Tensor(const vector<int>& dim)
        :dim(dim),vd(Prod(dim)),data(vd.data()) {}
    Tensor(const T* data, const vector<int>& dim)
        :dim(dim),data(data) {}

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
        if (size()!=t2.size())
            throw std::invalid_argument("Tensor::operator-=");

        VecMinusInplace(data,t2.data,size());
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
        vector<int> dim2(2);
        dim2[0]=std::accumulate(&dim[0],&dim[splitPos],1,std::multiplies<int>());
        dim2[1]=std::accumulate(dim.begin()+splitPos,dim.end(),1,std::multiplies<int>());
        return Tensor(data,dim2);
    }
    Tensor ReShape(vector<int> splitPos) const
    {
        vector<int> dim2(splitPos.size()+1);
        uint p=0,i;
        for(i=0;i<splitPos.size();i++)
        {
            dim2[i]=std::accumulate(&dim[p],&dim[splitPos[i]],1,std::multiplies<int>());
            p=dim[splitPos[i]];
        }
        dim2[i]=std::accumulate(dim.begin()+p,dim.end(),1,std::multiplies<int>());
        return Tensor(data,dim2);
    }

    Tensor operator*(const Tensor& t2) const
    {
        vector<int> dim_r;
        for(int i=0;i<dim.size()-1;i++)
            dim_r.push_back( dim[i] );
        for(int i=1;i<t2.dim.size();i++)
            dim_r.push_back( t2.dim[i] );
        Tensor r(dim_r);

        Tensor m1=this->ReShape(dim.size()-1);  //Matrix operation
        Tensor m2=t2.ReShape(1);
        Tensor mr=r.ReShape(dim.size()-1);
        MatMul(m1.data,m2.data,mr.data,m1.dim[0],m1.dim[1],m2.dim[1]);

        return r;
    }

    friend T Dot(const Tensor& t1,const Tensor& t2)
    {
        return VecDot(t1.data,t2.data,t1.size());
    }
};




#include"tensor.cpp"   // implementations

#endif // TENSOR_H
