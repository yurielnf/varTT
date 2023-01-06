#include "supertensor.h"

//#define EIGEN_USE_MKL_ALL
//#define EIGEN_USE_LAPACKE
//#define EIGEN_USE_BLAS

#include <Eigen/Core>
#include<Spectra/SymEigsSolver.h>
#include<iostream>

using namespace Spectra;
using namespace std;

EigenSystem0<TensorD> DiagonalizeArn(SuperTensor H, const TensorD& wf, int nIter, double tol)
{
    int ncv_default=min(51,nIter);
    int nev=min(1,wf.size()-1);
    while (ncv_default<=nIter)
    {
        int ncv=min(max(2*nev+1,ncv_default),wf.size());
        Spectra::SymEigsSolver<SuperTensor> eigs(H, nev, ncv);
        eigs.init(wf.data());
        eigs.compute(Spectra::SortRule::SmallestAlge, nIter/ncv, tol, Spectra::SortRule::SmallestAlge);
        if(eigs.info() == CompInfo::Successful)
        {
            double eval=eigs.eigenvalues()(0);
            auto v0=eigs.eigenvectors(1);
            stdvec x(v0.data(),v0.data()+v0.rows());
            auto  evec=TensorD(wf.dim,x);
            //        std::cout<<eval<< " Eigenvalues found:\n" << eigs.eigenvalues() << std::endl;
            return {eval,evec,
                        int(eigs.num_operations())};
        }
        ncv_default*=2;
    }
    throw std::runtime_error("Spectra::diagonalize info=");
}

