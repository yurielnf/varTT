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

template<class T> class Tensor;
template<class T> class TensorRef;
template<class T> class TensorCRef;
template<class T> class TensorRef2;
template<class Tensor> struct TensorNotation;

using TensorD=Tensor<double>;


template<class T>
struct Tensor
{
    typedef std::vector<T> C;

    Index  dim;
private:
    int n=0;
    std::vector<T> _vec;
    T* mem=nullptr;
public:
    T* data() {return mem==nullptr?_vec.data():mem;}
    const T* data() const {return mem==nullptr?_vec.data():mem;}
    int size() const {return n;}

    C vec() const {return C(data(),data()+size());}

    Tensor() {}
    Tensor(const Index& dim)
        :dim(dim),n(Prod(dim)),_vec(n) {}
    Tensor(const Index& dim,const std::vector<T>& vec)
        :dim(dim),n(Prod(dim)),_vec(vec)
    {}
    Tensor(const Index& dim,T* dat)
        :dim(dim),n(Prod(dim)),mem(dat)
    {}

    int rank() const {return dim.size();}

    void FillZeros()
    {
        VecFillZeros(data(),size());
    }
    void FillRandu()
    {
        VecFillRandu(data(),size());
    }
    void FillEye(int splitPos)
    {
        auto mt=ReShape(splitPos);
        if (mt.dim[0]!=mt.dim[1]) throw std::invalid_argument("FillEye");
        MatFillEye(data(),mt.dim[0]);
    }

    T& operator[](const Index& id)
    {
        return (*this)[Offset(id,dim)];
    }
    const T& operator[](const Index& id) const
    {
        return (*this)[Offset(id,dim)];
    }
    T& operator[](int i)
    {
        return data()[i];
    }
    const T& operator[](int i) const
    {
        return data()[i];
    }

    TensorNotation<Tensor&> operator()(std::string id)
    {
        return {*this,id};
    }
    TensorNotation<const Tensor&> operator()  (std::string id) const
    {
        return {*this,id};
    }

    void Save(std::ostream& out) const
    {
        for(int x:dim) out<<x<<" ";
        out<<"\n";
        VecSave(data(),size(),out);
    }
    void Save(std::string filename) const
    {
        std::ofstream out(filename); Save(out);
    }
    void Load(std::istream& in)
    {
        for(int x:dim) in>>x;
        VecLoad(data(),size(),in);
    }
    void Load(std::string filename)
    {
        std::ifstream in(filename);  Load(in);
    }

    void operator*=(T c)
    {
        VecProd(data(),size(),c);
    }
    void operator+=(const Tensor& t2)
    {
        if (dim!=t2.dim)
            throw std::invalid_argument("Tensor::operator-= incompatible");

        VecPlusInplace(data(),t2.data(),size());
    }
    void operator-=(const Tensor& t2)
    {
        if (dim!=t2.dim)
            throw std::invalid_argument("Tensor::operator-=incompatible");

        VecMinusInplace(data(),t2.data(),size());
    }

    Tensor operator-() const
    {
        Tensor y=*this;
        VecNegativeInplace(y.data(),size());
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

    Tensor ReShape(int splitPos)
    {
        auto dim_v=SplitIndex(dim,splitPos);
        return {{ Prod(dim_v[0]), Prod(dim_v[1])}, data()};
    }
    Tensor ReShape(int splitPos) const
    {
        auto dim_v=SplitIndex(dim,splitPos);
        return {{ Prod(dim_v[0]), Prod(dim_v[1])}, const_cast<T*>(data())};
    }

    Tensor ReShape(std::vector<int> splitPos)
    {
        auto dim_v=SplitIndex(dim,splitPos);
        Index dimr(dim_v.size());
        for(uint i=0;i<dim_v.size();i++)
            dimr[i]=Prod(dim_v[i]);
        return { dimr,data() };
    }
    Tensor ReShape(std::vector<int> splitPos) const
    {
        auto dim_v=SplitIndex(dim,splitPos);
        Index dimr(dim_v.size());
        for(uint i=0;i<dim_v.size();i++)
            dimr[i]=Prod(dim_v[i]);
        return { dimr,const_cast<T*>(data()) };
    }
    Tensor Subtensor(int i)
    {
        Tensor A=ReShape(rank()-1);
        Index dimr=dim;
        dimr.pop_back();
        return { dimr,data()+A.dim[0]*i };
    }
    Tensor Subtensor(int i) const
    {
        const Tensor A=ReShape(rank()-1);
        Index dimr=dim;
        dimr.pop_back();
        return { dimr,const_cast<T*>( data()+A.dim[0]*i) };
    }

    Tensor DiagMat() const
    {
        int n=size();
        Tensor t2({n,n});
        t2.FillZeros();
        for(int i=0;i<n;i++)
            t2[{i,i}]=data()[i];
        return t2;
    }

    Tensor Reorder(std::vector<int> posMap) const
    {
        Tensor t(IndexReorder(dim,posMap));
        for(int i=0;i<t.size();i++)
        {
//            int im=Offset(IndexReorder(ToIndex(i,dim),posMap),t.dim) ;
            int im=Offset(ToIndex(i,t.dim),dim,posMap);
            t[i]=(*this)[im];
        }
        return t;
    }
    Tensor Reorder(std::string ini,std::string fin) const
    {
        if (ini==fin) return *this;
        return Reorder(Permutation(ini,fin));
    }
    Tensor t() const
    {
        Index dimr=dim;
        std::swap(dimr.front(),dimr.back());
        const Tensor A=ReShape({1,rank()-1});
        return { dimr, A.Reorder("ijk","kji").vec() };
//        R("kji")=A("ijk");                                            //<---- TODO
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
        MatTranspose(m1.data(),t2.data(),m1.dim[0],m1.dim[1]);
        return t2;
    }
    Tensor operator*(const Tensor& t2) const
    {
        return Multiply(t2,1);
    }
    Tensor Multiply(const Tensor& t2, int nIdCommon) const
    {
        const Tensor& t1=*this;
        Index dim_r=IndexMul(t1.dim,t2.dim,nIdCommon);
        Tensor t3(dim_r);

        auto m1=t1.ReShape(t1.rank()-nIdCommon);  //Matrix operation
        auto m2=t2.ReShape(nIdCommon);
        auto m3=t3.ReShape(t1.rank()-nIdCommon);
        MatMul(m1.data(),m2.data(),m3.data(),m1.dim[0],m1.dim[1],m2.dim[1]);
        return t3;
    }

    friend Tensor DirectSum(const Tensor& t1,const Tensor& t2,bool left)
    {
        if (t1.rank()!=t2.rank())
            throw std::invalid_argument("TensorSum incompatible rank");

        Tensor A=t1.ReShape({1,t1.rank()-1}).Reorder("ijk","ikj");
        Tensor B=t2.ReShape({1,t1.rank()-1}).Reorder("ijk","ikj");
        if (A.dim[2]!=B.dim[2])
            throw std::invalid_argument("TensorSum incompatible inner index");
        Index dimc=A.dim;
        dimc[0]=left ? 1 : A.dim[0]+B.dim[0];
        dimc[1]=left ? A.dim[1]+B.dim[1] : 1;
        Tensor C(dimc);
        for(int i=0;i<C.dim[2];i++)
        {
            Tensor As=A.Subtensor(i);
            Tensor Bs=B.Subtensor(i);
            Tensor Cs=C.Subtensor(i);
            std::copy_n(As.data(),As.size(),Cs.data());
            std::copy_n(Bs.data(),Bs.size(),Cs.data()+As.size());
        }

        Index dimr=t1.dim;
        dimr.front()=C.dim[0];
        dimr.back()=C.dim[1];
        return { dimr,C.Reorder("ikj","ijk").vec() };
    }

//    friend Tensor DirectSum(const Tensor& t1,const Tensor& t2,bool left)
//    {
//        if (t1.rank()!=t2.rank())
//            throw std::invalid_argument("TensorSum incompatible rank");

//        Tensor A=t1.ReShape({1,t1.rank()-1});
//        Tensor B=t2.ReShape({1,t1.rank()-1});
//        if (A.dim[1]!=B.dim[1])
//            throw std::invalid_argument("TensorSum incompatible inner index");
//        Index dimr=t1.dim;
//        dimr.front()=left ? 1 : t1.dim.front()+t2.dim.front();
//        dimr.back()=left ? t1.dim.back()+t2.dim.back() : 1;
//        Tensor tr(dimr);
//        Tensor C=tr.ReShape({1,t1.rank()-1});
//        C.FillZeros();
//        for(int i=0;i<A.dim[0];i++)
//            for(int j=0;j<A.dim[1];j++)
//                for(int k=0;k<A.dim[2];k++)
//                    C[{i,j,k}]=A[{i,j,k}];
//        for(int i=0;i<B.dim[0];i++)
//            for(int j=0;j<B.dim[1];j++)
//                for(int k=0;k<B.dim[2];k++)
//                    if (left)
//                        C[{i,j,k+A.dim[2]}]=B[{i,j,k}];
//                    else
//                        C[{i+A.dim[0],j,k}]=B[{i,j,k}];

//        return tr;
//    }
    friend Tensor DirectSum(const Tensor& t1,const Tensor& t2)
    {
        if (t1.rank()!=t2.rank())
            throw std::invalid_argument("TensorSum incompatible rank");

        Tensor A=t1.ReShape({1,t1.rank()-1});
        Tensor B=t2.ReShape({1,t1.rank()-1});
        if (A.dim[1]!=B.dim[1])
            throw std::invalid_argument("TensorSum incompatible inner index");
        Index dimr=t1.dim;
        dimr.front()=t1.dim.front()+t2.dim.front();
        dimr.back()=t1.dim.back()+t2.dim.back();
        Tensor tr(dimr);
        Tensor C=tr.ReShape({1,t1.rank()-1});
        C.FillZeros();
        for(int i=0;i<A.dim[0];i++)
            for(int j=0;j<A.dim[1];j++)
                for(int k=0;k<A.dim[2];k++)
                    C[{i,j,k}]=A[{i,j,k}];
        for(int i=0;i<B.dim[0];i++)
            for(int j=0;j<B.dim[1];j++)
                for(int k=0;k<B.dim[2];k++)
                    C[{i+A.dim[0],j,k+A.dim[2]}]=B[{i,j,k}];

        return tr;
    }
    //---- friends ----

    friend bool operator==(const Tensor& t1,const Tensor& t2) //esta mal
    {
        return ( t1.dim==t2.dim && t1.vec()==t2.vec() );
    }
    friend bool operator!=(const Tensor& t1,const Tensor& t2)
    {
        return !(t1==t2);
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
        return VecNorm(t.data(),t.size());
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
//    friend std::vector<Tensor> SVDecomposition(const Tensor& t,int splitPos) //M=U*S*Vt
//    {
//        auto mt=t.ReShape(splitPos);
//        int n=std::min(mt.dim[0],mt.dim[1]);
//        auto dimUV=SplitIndex(t.dim,splitPos);
//        dimUV[0].push_back(n);    //U dimension
//        dimUV[1].insert(dimUV[1].begin(),n);    //V dimension
//        auto U=Tensor(dimUV[0]);   //U.FillZeros();
//        auto S=Tensor({n,n});      //S.FillZeros();
//        auto Vt=Tensor(dimUV[1]);  //Vt.FillZeros();
//        MatSVD(mt.data(),mt.dim[0],mt.dim[1],U.data(),S.data(),Vt.data());
//        return {U,S,Vt};
//    }
    friend std::vector<Tensor> SVDecomposition(const Tensor& t,int splitPos) //M=U*S*Vt
    {
        auto mt=t.ReShape(splitPos);
        int n=std::min(mt.dim[0],mt.dim[1]);
        auto dimUV=SplitIndex(t.dim,splitPos);
        dimUV[0].push_back(n);    //U dimension
        dimUV[1].push_back(n);    //V dimension
        auto U=Tensor(dimUV[0]);
        auto S=Tensor({n});
        auto V=Tensor(dimUV[1]);
        MatSVD(mt.data(),mt.dim[0],mt.dim[1],U.data(),S.data(),V.data());
        return {U,S.DiagMat(),V.Transpose(V.rank()-1)};
    }
};


template<class Tensor>
struct TensorNotation
{
    typedef typename std::decay<Tensor>::type _Tensor;

    Tensor t;
    std::string id;

    TensorNotation(Tensor t,std::string id)
        :t(t),id(id) {}

    template<class Tensor2>
    TensorNotation& operator=(const TensorNotation<Tensor2>& tn)
    {
        t=tn.t.Reorder(tn.id,id);
        return *this;
    }
    template<class Tensor2>
    TensorNotation<_Tensor> operator*(const TensorNotation<Tensor2>& tn)
    {
        auto ids=SortForMultiply(id,tn.id);
        auto t1=   t.Reorder(id   ,ids[0]);
        auto t2=tn.t.Reorder(tn.id,ids[1]);
        int nc=ids[3].length();
        auto t3=t1.Multiply(t2,nc);
        return {t3, ids[2]};
    }
};


//--------------------------------- other friends --------------------------------------

template<class T>
Tensor<T> operator*(const Tensor<T>& t,std::vector<const Tensor<T>*> transfer)
{
    Tensor<T> ts;
    const Tensor<T> &t0=*transfer[0];
    const Tensor<T> &t1=*transfer[1];
    const Tensor<T> &t2=*transfer[2];
    ts("IJK")=t0("ipI")*t("ijk")*t2("kqK")*t1("jpqJ");
    return ts;
}

template<class T>
Tensor<T> operator*(std::vector<const Tensor<T>*> transfer,const Tensor<T>& t)
{
    Tensor<T> ts;
    const Tensor<T> &t0=*transfer[0];
    const Tensor<T> &t1=*transfer[1];
    const Tensor<T> &t2=*transfer[2];
    ts("IJK")=t0("Ipi")*t("ijk")*t2("Kqk")*t1("Jpqj");
    return ts;
}



#include"tensor.cpp"   // implementations

#endif // TENSOR_H
