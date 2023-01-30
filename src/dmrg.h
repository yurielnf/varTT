#ifndef DMRG_H
#define DMRG_H

#include "dmrg1_wse_gs.h"
#include "dmrg_krylov_gs.h"

#include<iostream>
#include<armadillo>


struct DMRG {

    int m=1;            //< dmrg bond dimension
    int nsweep=1;
    int nIterMaxLanczos=64;
    double toldiag=1e-12;

    double energy=0;
    MPO ham;
    MPS gs;

    MPO NParticle() const
    {
        int L=ham.length;
        int m=4;
        MPSSum npart(m,MatSVDFixedTol(1e-13));
        for(int i=0;i<L; i++)
            npart += Fermi(i,L,true)*Fermi(i,L,false) ;
        return npart.toMPS();
    }

    void runDMRG0(const MPO& ham)
    {
        ham.PrintSizes();
        auto nop=NParticle();
        DMRG_krylov_gs sol(ham,m,1);
        sol.tol_diag=toldiag;
        sol.nIterMax=nIterMaxLanczos;
        for(int k=0;k<nsweep;k++)
        {
            std::cout<<"sweep "<<k+1<<" --------------------------------------\n";
            sol.DoIt_gs();
            sol.DoIt_res();
            sol.Print();

            Superblock np({&sol.gs[0],&nop,&sol.gs[0]});
            std::cout<<" nT="<<np.value()<<std::endl;
        }
        energy=sol.eval[0];
        gs=sol.gs[0];
    }

    void runDMRG1(const MPO& ham)
    {
        int len=ham.length;
        ham.PrintSizes();
        auto nop=NParticle();
        DMRG1_wse_gs sol(ham,m,1);
        sol.tol_diag=toldiag;
        sol.nIterMax=nIterMaxLanczos;
        sol.gs.decomposer=MatSVDFixedTol(sol.tol_diag);
        for(int k=0;k<nsweep;k++)
        {
            std::cout<<"sweep "<<k+1<<" --------------------------------------\n";
            for(auto p : MPS::SweepPosSec(len))
            {
                sol.SetPos(p);
                if (k+1==nsweep) sol.SolveNoWSE(false);
                else sol.Solve(false);
                sol.Print();
            }

            Superblock np({&sol.gs,&nop,&sol.gs});
            std::cout<<" nT="<<np.value()<<std::endl;
        }
        energy=sol.sb.value();
        gs=sol.gs;
    }

    arma::mat CalculateCiCj()
    {
        int N=gs.length;
        arma::mat cicj(N,N);
        for(int i=0;i<N;i++)
            for(int j=0;j<N;j++) {
                MPO cc=Fermi(i,N,true)*Fermi(j,N,false);
                cicj(i,j)=Superblock({&gs,&cc,&gs}).value();
            }
        return cicj;
    }
};


#endif // DMRG_H
