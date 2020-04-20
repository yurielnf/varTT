#ifndef SPECTRAWRAPPER_H
#define SPECTRAWRAPPER_H

#include <Eigen/Core>
#include<Spectra/SymEigsSolver.h>
#include<iostream>

using namespace Spectra;

template<class Ket>
struct EigenSystem
{
    double lambda0;
    Ket state;
    int iter;
    const Ket& GetState() const {return state;}

};

template<class Hamiltonian, class Ket>                                      //Portal method
EigenSystem<Ket> DiagonalizeArn(Hamiltonian H, Ket wf,int nIter,double tol)
{
    Spectra::SymEigsSolver<double, Spectra::SMALLEST_ALGE, Hamiltonian> eigs(&H, 4, 11);
    eigs.init(wf.data());
    eigs.compute();
    if(eigs.info() == Spectra::SUCCESSFUL)
    {
        double eval=eigs.eigenvalues()(0);
        auto  evec=Ket(wf.dim,eigs.eigenvectors(1).data()).Clone();
        return {eval,evec,
                    eigs.num_operations()};
        std::cout << "Eigenvalues found:\n" << eigs.eigenvalues() << std::endl;
    }
    else
        throw std::runtime_error("Spectra::diagonalize info="+std::to_string(eigs.info()));
}


#endif // SPECTRAWRAPPER_H
