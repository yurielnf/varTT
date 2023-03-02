#ifndef DMRG_H
#define DMRG_H

#include "dmrg1_wse_gs.h"
#include "dmrg_krylov_gs.h"
#include "superblock_corr.h"

#include<iostream>
#include<armadillo>
#include<variant>


struct DMRG_base {
    int m=64;            //< dmrg bond dimension
    int nIter_diag=64;
    double tol_diag=1e-12;
    bool use_arpack=false;

    DMRG_base(MPO const& ham_): ham(ham_) {}
    DMRG_base(MPO const& ham_, MPS const& gs_): ham(ham_), gs(gs_) {}

    double Expectation(MPO &O) { return  Superblock({&gs,&O,&gs}).value(); }

    double correlation(MPO& Oij, int i, int j)
    {
        SuperBlock_Corr sb(gs);
        return sb.value(Oij,i,j);
    }

    double sigma(int bond_dim)
    {
        double ener=Superblock({&gs,&ham,&gs}).value();
        MPO Heff=ham + MPOIdentity(ham.length, ham.at(0).dim[1])*(-ener);
        return MPO_MPS{Heff,gs}.toMPS(bond_dim).norm();
    }

    double H2(int bond_dim) { return std::pow(MPO_MPS{ham,gs}.toMPS(bond_dim).norm(), 2); }

    int sweep=0;
    double energy=0;
    MPO ham;
    MPS gs;
};

struct DMRG: public DMRG_base {
    DMRG(MPO const& ham_) : DMRG_base(ham_), sol(ham_,m,1) {}
    DMRG(MPO const& ham_, MPS const& gs_) : DMRG_base(ham_,gs_), sol(ham_,gs_,m) {}

    void iterate()
    {
        sol.tol_diag=tol_diag;
        sol.gs.m=m;
        sol.gs.decomposer=MatSVDAdaptative(tol_diag,m);
        sol.nIterMax=nIter_diag;


        for(auto p : MPS::SweepPosSec(ham.length))
        {
            sol.SetPos(p);
            sol.Solve(use_arpack);
        }

        energy=sol.sb.value();
        gs=sol.gs;
    }

private:
    DMRG1_wse_gs sol;
};

struct DMRG0: public DMRG_base {

    DMRG0(MPO const& ham_, int nKrylov=2) : DMRG_base(ham_), sol(ham_,m,nKrylov) {}
    DMRG0(MPO const& ham_, MPS const& gs_, int nKrylov=2) : DMRG_base(ham_,gs_), sol(ham_,gs_,m,nKrylov) {}

    void iterate()
    {
        static int c=0;

        sol.tol_diag=tol_diag;
        sol.gs[0].m=m;
        sol.gs[0].decomposer=MatSVDAdaptative(tol_diag,m);
        sol.nIterMax=nIter_diag;

            if (c++ > 0) sol.reset_states();
            sol.DoIt_gs();
            sol.DoIt_res();

        gs=sol.gs[0];
        energy=Superblock({&gs,&ham,&gs}).value();
    }

private:
    DMRG_krylov_gs sol;
};





#endif // DMRG_H
