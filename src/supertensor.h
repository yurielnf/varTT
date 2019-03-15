#ifndef SUPERTENSOR_H
#define SUPERTENSOR_H

#include<vector>
#include"tensor.h"

struct SuperTensor
{
    TensorD _A,_B;
    SuperTensor(const TensorD& A,const TensorD& B,
                const std::vector<TensorD>& C={})
        :_A(A),_B(B)
    {
        if (C.size()==1)
            _A("iJk")=_A("ijk")*C[0]("jJ");
    }
    TensorD operator*(const TensorD& psi) const
    {
        TensorD tr;
        if (_A.rank()==2)
            tr("kK")=_A("ik")*psi("iI")*_B("IK");
        else if (_A.rank()==3)
            tr("kK")=_A("ijk")*psi("iI")*_B("IjK");
        return tr;
    }
};

#endif // SUPERTENSOR_H
