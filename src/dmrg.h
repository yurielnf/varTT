#ifndef DMRG_H
#define DMRG_H

#include "dmrg1_wse_gs.h"
#include "dmrg_krylov_gs.h"

#include<iostream>
#include<armadillo>
#include<variant>


struct DMRG_base {
    int m=64;            //< dmrg bond dimension
    int nIter_diag=64;
    double tol_diag=1e-12;
    bool use_arpack=false;

    DMRG_base(MPO const& ham_): ham(ham_) { ham.PrintSizes("ham"); }

    double Expectation(MPO &O) { return  Superblock({&gs,&O,&gs}).value(); }

    int sweep=0;
    double energy=0;
    MPO ham;
    MPS gs;
};

struct DMRG: public DMRG_base {
    DMRG(MPO const& ham_) : DMRG_base(ham_), sol(ham_,m,1) {}

    void iterate()
    {
        sol.tol_diag=tol_diag;
        sol.gs.decomposer=MatSVDFixedTol(sol.tol_diag);
        sol.nIterMax=nIter_diag;

        std::cout<<"sweep "<<++sweep<<" --------------------------------------\n";
        for(auto p : MPS::SweepPosSec(ham.length))
        {
            sol.SetPos(p);
            sol.Solve(use_arpack);
            sol.Print();
        }

        energy=sol.sb.value();
        gs=sol.gs;
    }

private:
    DMRG1_wse_gs sol;
};

struct DMRG0: public DMRG_base {

    DMRG0(MPO const& ham_) : DMRG_base(ham_), sol(ham_,m,1) {}

    void iterate()
    {
        sol.tol_diag=tol_diag;
        sol.nIterMax=nIter_diag;
        {
            std::cout<<"sweep "<<++sweep<<" --------------------------------------\n";
            sol.DoIt_gs();
            sol.DoIt_res();
            sol.Print();
        }

        energy=sol.eval[0];
        gs=sol.gs[0];
    }
private:
    DMRG_krylov_gs sol;
};





#endif // DMRG_H
