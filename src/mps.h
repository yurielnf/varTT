#ifndef MPS_H
#define MPS_H

#include<vector>
#include"tensor.h"

class MPS
{
public:
    std::vector<TensorD> M;
    TensorD C=TensorD({1,1}, {1});
    double norm_n=1;                //norm(MPS)^(1/n)
    int pos=-1;
    double tol=1e-14;

    MPS(int n):M(n) {}

    void FillRandu()
    {
        for(TensorD &x:M) x.FillRandu();
    }

    void Canonicalize()
    {
        ExtractNorm();
        C=Tensor<double>({1,1},{1});
        while(pos<=M.size()/2) SweepRight();
        auto cC=C;
        C=Tensor<double>({1,1},{1});
        while(pos>M.size()/2) SweepLeft();
        C=cC*C;
    }
    void ExtractNorm()
    {
        for(TensorD& psi:M)
        {
            double nr=Norm(psi);
            if (nr<tol)
                throw std::logic_error("mps:ExtractNorm() null matrix");
            norm_n*=pow(nr,1.0/M.size());
            psi*=1.0/nr;
        }
    }
    void SweepRight()
    {
        pos++;
        auto psi=C*M[pos];
        auto usvt=SVDDecomposition(psi,2);
        M[pos]=usvt[0];
        C=usvt[1]*usvt[2];
    }
    void SweepLeft()
    {
        auto psi=M[pos]*C;
        auto usvt=SVDDecomposition(psi,1);
        M[pos]=usvt[2];
        C=usvt[0]*usvt[1];
        pos--;
    }

};

#endif // MPS_H
