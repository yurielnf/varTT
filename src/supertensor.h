#ifndef SUPERTENSOR_H
#define SUPERTENSOR_H

#include<vector>
#include"tensor.h"

struct SuperTensor
{
    TensorD _A,_B;
    SuperTensor(const TensorD& A,const TensorD& B,
                const std::vector<TensorD>& C={})
        :_B(B)
    {
        if (C.empty())
            _A=A;
        else
        {
            if (C[0].rank()==2)
                _A("iJk")=A("ijk")*C[0]("jJ");
            else if(C[0].rank()==4)
                _A("iaJbk")=A("ijk")*C[0]("jabJ");
            else
                throw std::invalid_argument("SuperTensor()");
        }
    }
    TensorD operator*(const TensorD& psi) const
    {
        TensorD tr;
        if (_A.rank()==2)
        {
            if (psi.rank()==2)
                tr("kK")=_A("ik")*psi("iI")*_B("IK");
            else if (psi.rank()==3)
                tr("kaK")=_A("ik")*psi("iaI")*_B("IK");
        }
        else if (_A.rank()==3)
            tr("kK")=_A("ijk")*psi("iI")*_B("IjK");
        else if(_A.rank()==5)
            tr("kbK")=_A("iajbk")*psi("iaI")*_B("IjK");
        else
            throw std::invalid_argument("SuperTensor*psi");
        return tr;
    }
};

#endif // SUPERTENSOR_H
