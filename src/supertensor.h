#ifndef SUPERTENSOR_H
#define SUPERTENSOR_H

#include<vector>
#include"tensor.h"

struct SuperTensor
{
    TensorD _A,_B;
    SuperTensor(const TensorD& A,const TensorD& C,const TensorD& B)
        :_A(A),_B(B)
    {
        _A("iJk")=_A("ijk")*C("jJ");
    }
    TensorD operator*(const TensorD& psi) const
    {
        TensorD tr;
        tr("kK")=_A("ijk")*psi("iI")*_B("IjK");
        return tr;
    }
};

#endif // SUPERTENSOR_H
