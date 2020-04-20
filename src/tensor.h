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

template<class T> struct Tensor;
template<class Tensor> struct TensorNotation;

using TensorD=Tensor<double>;


template<class T>
struct Tensor
{
    typedef std::vector<T> C;

    Index  dim,dim_prod;
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
        :dim(dim),dim_prod(DimProd(dim)),n(Prod(dim)),_vec(n) {}
    Tensor(const Index& dim,const std::vector<T>& vec)
        :dim(dim),dim_prod(DimProd(dim)),n(Prod(dim)),_vec(vec)
    {
        if ( (int)vec.size() !=n )
            throw std::invalid_argument("Tensor from vec incompatible");
    }
    Tensor(const Index& dim,T* dat)
        :dim(dim),dim_prod(DimProd(dim)),n(Prod(dim)),mem(dat) {}

    void redim(const Index &dim2)
    {
        if (dim!=dim2) return;
        if (mem!=nullptr)
            throw std::invalid_argument("redim of reshape");
        dim=dim2;
        dim_prod=DimProd(dim);
        n=Prod(dim);
        _vec.resize(n);
    }
    /*friend void swap(Tensor& t1,Tensor& t2)
    {
        std::swap(t1.dim,t2.dim);
        std::swap(t1.dim_prod,t2.dim_prod);
        std::swap(t1.n,t2.n);
        std::swap(t1._vec,t2._vec);
        std::swap(t1.mem,t2.mem);
    }

    Tensor(const Tensor& t)
        :dim(t.dim)
        ,dim_prod(t.dim_prod)
        ,n(t.n)
        ,_vec(t._vec)
        ,mem(t.mem)
    {}
    Tensor(Tensor&& t)
        :Tensor()
    { swap(*this,t); }

    Tensor& operator=(const Tensor& t)
    {
        Tensor t2=t;
        swap(*this,t2);
        return *this;
    }
    Tensor& operator=(Tensor&& t)
    {
        swap(*this,t);
        return *this;
    }*/

    Tensor Clone() const { return {dim,vec()}; }
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
        return (*this)[OffsetP(id,dim_prod)];
    }
    const T& operator[](const Index& id) const
    {
        return (*this)[OffsetP(id,dim_prod)];
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
        out<<rank()<<"\n";
        for(int x:dim) out<<x<<" ";
        VecSave(data(),size(),out,false);
    }
    void Load(std::istream& in)
    {
        int r;
        in>>r;
        dim.resize(r);
        for(int& x:dim) in>>x;
        *this=Tensor(dim);
        in.ignore(1);
        VecLoad(data(),size(),in,false);
    }
    void Save(std::string filename) const
    {
        std::ofstream out(filename); Save(out);
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
        if(rank()==0) {*this=t2; return;}
        if (dim!=t2.dim)
            throw std::invalid_argument("Tensor::operator+= incompatible");

        VecPlusInplace(data(),t2.data(),size());
    }
    void pexa(const Tensor& t,double a) //t+=t*a
    {
        if(rank()==0) {*this=t; return;}
        if (dim!=t.dim)
            throw std::invalid_argument("Tensor::operator+=x a incompatible");

        Vec_xa_Inplace(data(),t.data(),a,size());
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

//    Tensor Reorder(std::vector<int> posMap) const
//    {
//        Tensor t(IndexReorder(dim,posMap));
//        for(int i=0;i<t.size();i++)
//        {
////            int im=Offset(IndexReorder(ToIndex(i,dim),posMap),t.dim) ;
//            int im=Offset(ToIndex(i,t.dim),dim,posMap);
//            t[i]=(*this)[im];
//        }
//        return t;
//    }
    Tensor Reorder(const std::vector<int>& posMap) const
    {
        Tensor t(IndexReorder(dim,posMap));
        Index id(t.dim.size(),0);               // <-----------------  to do: manual id2 for im
        int pos=0;
        for(int i=0;i<t.size();i++)
        {
//            int im=Offset(IndexReorder(ToIndex(i,dim),posMap),t.dim) ;
//            id=ToIndex(i,t.dim);
            int im=OffsetP(id,dim_prod,posMap);
            t[i]=(*this)[im];

//            if (id2!=id) throw std::runtime_error("Reorder: id!=id2");
            id[pos]++;
            if (id[pos]==t.dim[pos])
            {
                while (id[pos]==t.dim[pos] && pos<t.rank())
                {
                    id[pos]=0;
                    pos++;
                    if(pos<t.rank()) id[pos]++;
                }
                pos=0;
            }
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
    void Multiply(const Tensor& t2, int nIdCommon,Tensor &t3) const
    {
        const Tensor& t1=*this;
        Index dim_r=IndexMul(t1.dim,t2.dim,nIdCommon);
        //Tensor t3(dim_r);
        if (t3.dim!=dim_r)
            throw std::invalid_argument("Tensor::Multiply");

        auto m1=t1.ReShape(t1.rank()-nIdCommon);  //Matrix operation
        auto m2=t2.ReShape(nIdCommon);
        auto m3=t3.ReShape(t1.rank()-nIdCommon);
        MatMul(m1.data(),m2.data(),m3.data(),m1.dim[0],m1.dim[1],m2.dim[1]);
        //return t3;
    }
    Tensor Multiply(const Tensor& t2, int nIdCommon) const
    {
        const Tensor& t1=*this;
        Index dim_r=IndexMul(t1.dim,t2.dim,nIdCommon);
        Tensor t3(dim_r);
        Multiply(t2,nIdCommon,t3);
        return t3;
    }

    void MultiplyT(const Tensor& t2,int splitPos,int nIdCommon,Tensor &t3) const
    // t1.Mult(t2.t(splitPos),nIdC)
    {
        const Tensor& t1=*this;
        auto dim2=TransposeIndex(t2.dim,splitPos);
        Index dim_r=IndexMul(t1.dim,dim2,nIdCommon);
        //Tensor t3(dim_r);
        if (t3.dim!=dim_r)
            throw std::invalid_argument("Tensor::MultiplyT");

        auto m1=t1.ReShape(t1.rank()-nIdCommon);  //Matrix operation
        auto m2=t2.ReShape(t2.rank()-nIdCommon);
        auto m3=t3.ReShape(t1.rank()-nIdCommon);
        MatMulT(m1.data(),m2.data(),m3.data(),m1.dim[0],m1.dim[1],m2.dim[0]);
        //return t3;
    }
    void TMultiply(const Tensor& t2,int splitPos,int nIdCommon,Tensor &t3) const
    // t1.Mult(t2.t(splitPos),nIdC)
    {
        const Tensor& t1=*this;
        auto dim1=TransposeIndex(t1.dim,splitPos);
        Index dim_r=IndexMul(dim1,t2.dim,nIdCommon);
        //Tensor t3(dim_r);
        if (t3.dim!=dim_r)
            throw std::invalid_argument("Tensor::TMultiply");

        auto m1=t1.ReShape(nIdCommon);  //Matrix operation
        auto m2=t2.ReShape(nIdCommon);
        auto m3=t3.ReShape(t1.rank()-nIdCommon);
        MatTMul(m1.data(),m2.data(),m3.data(),m1.dim[0],m1.dim[1],m2.dim[1]);
        //return t3;
    }

//    friend Tensor DirectSum(const Tensor& t1,const Tensor& t2,bool left)
//    {
//        if (t1.rank()!=t2.rank())
//            throw std::invalid_argument("TensorSum incompatible rank");

//        Tensor A=t1.ReShape({1,t1.rank()-1}).Reorder("ijk","ikj");
//        Tensor B=t2.ReShape({1,t1.rank()-1}).Reorder("ijk","ikj");
//        if (A.dim[2]!=B.dim[2])
//            throw std::invalid_argument("TensorSum incompatible inner index");
//        Index dimc=A.dim;
//        dimc[0]=left ? 1 : A.dim[0]+B.dim[0];
//        dimc[1]=left ? A.dim[1]+B.dim[1] : 1;
//        Tensor C(dimc);
//        for(int i=0;i<C.dim[2];i++)
//        {
//            Tensor As=A.Subtensor(i);
//            Tensor Bs=B.Subtensor(i);
//            Tensor Cs=C.Subtensor(i);
//            std::copy_n(As.data(),As.size(),Cs.data());
//            std::copy_n(Bs.data(),Bs.size(),Cs.data()+As.size());
//        }

//        Index dimr=t1.dim;
//        dimr.front()=C.dim[0];
//        dimr.back()=C.dim[1];
//        return { dimr,C.Reorder("ikj","ijk").vec() };
//    }
    friend Tensor DirectSum(const Tensor& t1,const Tensor& t2,bool left)
    {
//        if (t1.rank()!=t2.rank())
//            throw std::invalid_argument("TensorSum incompatible rank");

        Tensor A=t1.ReShape({1,t1.rank()-1});
        Tensor B=t2.ReShape({1,t1.rank()-1});
        if (A.dim[1]!=B.dim[1])
            throw std::invalid_argument("TensorSum incompatible inner index");
        Index dimr=t1.dim,delta={0,0,0};
        if (!left)
        {
            if (A.dim[2]!=B.dim[2])
                throw std::invalid_argument("TensorSum incompatible right index");
            dimr.front()=t1.dim.front()+t2.dim.front();
            delta[0]=A.dim[0];
        }
        else
        {
            if (A.dim[0]!=B.dim[0])
                throw std::invalid_argument("TensorSum incompatible left index");
            dimr.back()=t1.dim.back()+t2.dim.back();
            delta[2]=A.dim[2];
        }
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
                    C[{i+delta[0],j,k+delta[2]}]=B[{i,j,k}];

        return tr;
    }
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
    std::array<Tensor,2> EigenDecomposition(int splitPos)
    {
        const Tensor& t=*this;
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
//        dimUV[1].push_back(n);    //V dimension
//        auto U=Tensor(dimUV[0]);
//        auto S=Tensor({n});
//        auto V=Tensor(dimUV[1]);
//        MatSVD(mt.data(),mt.dim[0],mt.dim[1],U.data(),S.data(),V.data());
//        return {U,S.DiagMat(),V.Transpose(V.rank()-1)};
//    }
    typedef std::function< std::array<stdvec,2>(
                                                bool is_right,
                                                const double*  X,
                                                int n1,int n2)> mat_decomp;
    std::array<Tensor,2> Decomposition(bool is_right,
                                       mat_decomp decomp) const
    {
        const Tensor &t=*this;
        int splitPos= is_right ? 1 : t.rank()-1;
        auto mt=t.ReShape(splitPos);

        auto uv=decomp(is_right,mt.data(),mt.dim[0],mt.dim[1]);
        int n=uv[0].size()/mt.dim[0];
        auto dimUV=SplitIndex(t.dim,splitPos);
        dimUV[0].push_back(n);                  //U dimension
        dimUV[1].insert(dimUV[1].begin(),n);    //V dimension
        auto U=Tensor(dimUV[0],uv[0]);
        auto V=Tensor(dimUV[1],uv[1]);
        return {U,V};
    }
//    std::array<Tensor,2> ChopMatDecomposition(bool is_right) const
//    {
//        for(int j=0;j<dim.back();j++)
//        {
//            auto ts=Subtensor(j);
//            double mc=*std::max_element(ts.data(),ts.data()+ts.n,
//                                        [](T a,T b){return std::abs(a)<std::abs(b);})
//        }
//    }
//    std::array<Tensor,2> ChopDecomposition(bool is_right) const
//    {
//        const Tensor &t=*this;
//        int splitPos= is_right ? 1 : t.rank()-1;
//        auto mt=t.ReShape(splitPos);

//        auto uv=ChopMatDecomposition(is_right);
//        int n=uv[0].size()/mt.dim[0];
//        auto dimUV=SplitIndex(t.dim,splitPos);
//        dimUV[0].push_back(n);    //U dimension
//        dimUV[1].insert(dimUV[1].begin(),n);    //V dimension
//        auto U=Tensor(dimUV[0],uv[0]);
//        auto V=Tensor(dimUV[1],uv[1]);
//        return {U,V};
//    }
};



template<class Tensor>
struct TensorNotation
{
    typedef typename std::decay<Tensor>::type _Tensor;

    Tensor t;
    std::string id;

    TensorNotation(Tensor t,std::string id)
        :t(t),id(id)
    {
        if (t.rank()>0 && t.rank()!=(int)id.size())
            throw std::invalid_argument(
                    "TensorNotation rank != string length: "+id
                    +" != "+std::to_string(t.rank()));
    }

    template<class Tensor2>
    TensorNotation& operator=(const TensorNotation<Tensor2>& tn)
    {
        if (id==tn.id)
            t=tn.t;
        else
            t=tn.t.Reorder(tn.id,id);
        return *this;
    }
    template<class Tensor2>
    TensorNotation<_Tensor> operator*(const TensorNotation<Tensor2>& tn)
    {
        auto ids=SortForMultiply(id,tn.id);
        int nc=ids[3].length();
        _Tensor t3;
        if (id==ids[0] && tn.id==ids[1])
            t3=t.Multiply(tn.t,nc);
        else if(id==ids[0])
        {
            auto t2=tn.t.Reorder(tn.id,ids[1]);
            t3=t.Multiply(t2,nc);
        }
        else if(tn.id==ids[1])
        {
            auto t1=   t.Reorder(id   ,ids[0]);
            t3=t1.Multiply(tn.t,nc);
        }
        else
        {
            auto t1=   t.Reorder(id   ,ids[0]);
            auto t2=tn.t.Reorder(tn.id,ids[1]);
            t3=t1.Multiply(t2,nc);
        }
        return {t3, ids[2]};
    }
};


//--------------------------------- other friends --------------------------------------

template<class T=double>
std::vector<T> flat(const std::vector<Tensor<T>> &vt)
{
    std::vector<T> v;
    for(const Tensor<T>& x:vt)
        v.insert(v.end(),x.data(),x.data()+x.size());
    return v;
}

template<class T>
using TransferTensor=std::vector<const Tensor<T>*>;

/*
template<class T>
Tensor<T> operator*(const Tensor<T>& t,TransferTensor<T> transfer)
{
    Tensor<T> ts;
    const Tensor<T> &t0=*transfer[0];
    const Tensor<T> &t1=*transfer[1];
    if(transfer.size()==2)
        //ts("IJ")=t0("ipI")*t("ij")*t1("jpJ");
        ts=t0.Transpose(2).Multiply(t*t1,2);
    else if (transfer.size()==3)
    {
        const Tensor<T> &t2=*transfer[2];
//        ts("IJK")=t0("ipI")*t("ijk")*t2("kqK")*t1("jpqJ");
        auto a=t*t2; //ijqK
        auto b=a.Transpose(3).Multiply(t1.Reorder("jpqJ","jqpJ"),2); //KipJ
        ts=t0.Transpose(2).Multiply(b.Transpose(1),2);
    }
    return ts;
}*/

template<class T>
Tensor<T> operator*(const Tensor<T>& t,TransferTensor<T> transfer)
{
    Tensor<T> ts;
    const Tensor<T> &t0=*transfer[0];
    const Tensor<T> &t1=*transfer[1];
    if(transfer.size()==2)
    {
        //ts("IJ")=t0("ipI")*t("ij")*t1("jpJ");
        //ts("JI")=t1("jpJ")*( t("ji")*t0("ipI") );
        ts=TensorD({t1.dim.back(),t0.dim.back()});
        t1.TMultiply(t*t0,2,2,ts);
    }
    else if (transfer.size()==3)
    {
        const Tensor<T> &t2=*transfer[2];
//        ts("IJK")=t0("ipI")*t("ijk")*t2("kqK")*t1("jpqJ");
//        ts("KJI")=t2("kqK")*(  t("kji")*t0("ipI")*t1("jpqJ")  );
        auto a=t*t0; //kjpI
        TensorD b=TensorD({a.dim[0], t1.dim[2], t1.dim[3], a.dim[3]}); //kqJI
        for(int i=0;i<b.dim.back();i++)
        {
            auto bi=b.Subtensor(i);
            a.Subtensor(i).Multiply(t1,2,bi);
        }
        ts=TensorD({t2.dim.back(),t1.dim.back(),t0.dim.back()});
        t2.TMultiply(b,2,2,ts); //KJI
    }
    return ts;
}

/*
template<class T>
Tensor<T> operator*(TransferTensor<T> transfer,const Tensor<T>& t)
{
    Tensor<T> ts;
    const Tensor<T> &t0=*transfer[0];
    const Tensor<T> &t1=*transfer[1];
    if(transfer.size()==2)
//        ts("IJ")=t0("Ipi")*t("ij")*t1("Jpj");
        ts=(t0*t).Multiply(t1.Transpose(1),2);
    else if (transfer.size()==3)
    {
        const Tensor<T> &t2=*transfer[2];
//        ts("IJK")=t0("Ipi")*t("ijk")*t2("Kqk")*t1("Jpqj");
        auto a=t0*t; //Ipjk
        auto b=a.Transpose(3).Multiply(t1.Reorder("Jpqj","pjJq"),2);     //kIJq
        ts=b.Transpose(1).Multiply(t2.Transpose(1),2);  //IJK
    }
    return ts;
}*/

template<class T>
Tensor<T> operator*(TransferTensor<T> transfer,const Tensor<T>& t)
{
    Tensor<T> ts;
    const Tensor<T> &t0=*transfer[0];
    const Tensor<T> &t1=*transfer[1];
    if(transfer.size()==2)
    {
//        ts("IJ")=t0("Ipi")*t("ij")*t1("Jpj");
//        ts("JI")=t1("Jpj")*t("ji")*t0("Ipi");
        ts=TensorD({t1.dim[0],t0.dim[0]});
        (t1*t).MultiplyT(t0,1,2,ts);
    }
    else if (transfer.size()==3)
    {
        const Tensor<T> &t2=*transfer[2];
//        ts("IJK")=t0("Ipi")*t("ijk")*t2("Kqk")*t1("Jpqj");    //viejo
//        ts("KJI")=t2("Kqk")*t("kji")*t1("Jpqj")*t0("Ipi");

        auto a=t2*t; //Kqji
        auto b=TensorD({a.dim[0],t1.dim[0],t1.dim[1],a.dim[3]}); //KJpi
        for(int i=0;i<b.dim.back();i++)
        {
            auto bi=b.Subtensor(i);
            a.Subtensor(i).MultiplyT(t1,2,2,bi);
        }
        ts=TensorD({t2.dim[0],t1.dim[0],t0.dim[0]});
        b.MultiplyT(t0,1,2,ts);  //KJI
    }
    return ts;
}



#include"tensor.cpp"   // implementations

#endif // TENSOR_H
