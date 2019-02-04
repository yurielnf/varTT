#ifndef MPS_H
#define MPS_H

#include<vector>
#include"tensor.h"

class MPS
{
public:
    std::vector<TensorD> M;
    TensorD C;
    double norm_n; // norm(M)^(1/n)
    int pos=0;

    MPS(int n)
        :M(n),C({1,1}, {1})
    {}

    void FillRandu()
    {
        for(TensorD &x:M) x.FillRandu();
    }

    void Canonicalize()
    {
        C=Tensor<double>({1,1},{1});
        for(pos=0;pos<=M.size()/2;pos++)
            SweepRight();
    }
    void SweepRight();

};

#endif // MPS_H
