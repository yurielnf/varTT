#ifndef HAMHALL_H
#define HAMHALL_H

#include "model/fermionic.h"

#include<armadillo>


struct IRLM {
    int L=100;
    double t=0.5;
    double V=0.15;
    double U=-0.5;

    Fermionic model() const
    {
        // Kinetic energy TB Hamiltonian
        arma::mat K(L,L,arma::fill::zeros);
        for(auto i=1; i<L-1; i++)
            K(i,i+1)=K(i+1,i)=t;
        K(0,1)=K(1,0)=V;

        // U ni nj
        arma::mat Umat(L,L,arma::fill::zeros);
        Umat(0,1)=U;
        K(0,0)=K(1,1)=-U/2;


        { // Diagonalize the bath
            arma::vec ek;
            arma::mat R;
            arma::eig_sym( ek, R, K.submat(2,2,L-1,L-1) );
            arma::uvec iek=arma::sort_index( arma::abs(ek) );
            arma::mat Rfull(L,L,arma::fill::eye);
            arma::mat Rr=R.cols(iek);
            Rfull.submat(2,2,L-1,L-1)=Rr;
            K=Rfull.t()*K*Rfull;
        }
        return Fermionic(K,Umat);
    }
};





#endif // HAMHALL_H
