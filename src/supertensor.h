#ifndef SUPERTENSOR_H
#define SUPERTENSOR_H

#include<vector>
#include"tensor.h"

struct SuperTensor
{
    TensorD A,B;
    Index ix;
    std::vector<TensorD> C;
    SuperTensor(const TensorD& A_,const TensorD& B_,
                const Index& ix,
                const std::vector<TensorD>& C_={})
        :A(A_)
        ,B(B_)
        ,ix(ix)
        ,C(C_)
    {
        if (!C.empty() && C[0].rank()==2)
        {   //A("kJi")=A("kji")*C[0]("jJ")
//            A=(A.Transpose(2)*C[0]).Transpose(1);
            for(int i=0;i<A.dim[2];i++)
            {
                auto ai=A.Subtensor(i);
                ai.Clone().Multiply(C[0],1,ai);
            }
            C.clear();
        }
//        a=TensorD(IndexMul(ix,B.dim,1)); //x*B
//        if (A.rank()==3 && !C.empty())
//            b=TensorD({a.dim[0],C[0].dim[0],C[0].dim[1],a.dim[3]}); //kjaI
    }
    int rows() const { return Prod(ix); }
    int cols() const { return Prod(ix); }
    void perform_op(const double *x_in, double *y_out) const
    {
        const TensorD x(ix,const_cast<double*>(x_in));
        TensorD y(ix,y_out);
        apply(x,y);
    }

    void apply(const TensorD& psi,TensorD& y) const
    {
        auto a=TensorD(IndexMul(psi.dim,B.dim,1));
        psi.Multiply(B,1,a);
        if (A.rank()==2)
        {
            // tr("iI")=A("ki")*psi("kK")*B("KI");
            // tr("iaI")=A("ki")*psi("kaK")*B("KI");
            A.TMultiply(a,1,1,y);
//            tr=A.Transpose(1)*psi*B;
        }
        else if (C.empty())
//            tr("iI")=A("kji")*psi("kK")*B("KjI");
            A.TMultiply(a,2,2,y);
//            tr=A.Transpose(2).Multiply(psi*B,2);
        else
//            tr("kbK")=A("kji")*( psi("kbK")*B("KJI")*C[0]("jabJ"));
        {            
            auto b=TensorD({a.dim[0],C[0].dim[0],C[0].dim[1],a.dim[3]}); //kjaI
            const auto &ct=C[0];
            for(int i=0;i<b.dim.back();i++)
            {
                auto bi=b.Subtensor(i);
                a.Subtensor(i).MultiplyT(ct,2,2,bi);
            }
            A.TMultiply(b,2,2,y);
//            tr=A.Transpose(2).Multiply(b,2); //iaI
        }
    }
    TensorD operator*(const TensorD &psi) const
    {
        auto dim2=psi.dim;
        dim2.front()=A.dim.back();
        dim2.back()=B.dim.back();
        auto y=TensorD(dim2);
        apply(psi,y);
        return y;
    }

//private:
//     mutable TensorD  a,b;
};

EigenSystem0<TensorD> DiagonalizeArn(SuperTensor H, const TensorD& wf, int nIter, double tol);


#endif // SUPERTENSOR_H
