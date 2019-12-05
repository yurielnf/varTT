#include"catch.hpp"
#include"parameters.h"
#include"dmrg_res_gs.h"
#include"dmrg1_res_gs.h"
#include"dmrg_jd_gs.h"
#include"dmrg1_jd_gs.h"

static double ExactEnergyTB(int L, int nPart,bool periodic)
{
    std::vector<double> evals(L);
    for(int k=0;k<L;k++)
    {
        double kf= periodic ? 2*M_PI*k/L: M_PI*(k+1)/(L+1);
        evals[k]=2*cos(kf);
    }
    std::sort(evals.begin(),evals.end());
    double sum=0;
    for(int k=0;k<nPart;k++)
        sum+=evals[k];
    return sum;
}

TEST_CASE( "dmrg0 tight-binding", "[dmrg_0_tb]" )
{
    srand(time(NULL));
    int len=10, m=200, mMax=200;
    std::cout<<std::setprecision(15);

    SECTION( "dmrg" )
    {
        auto op=HamTbAuto(len,false); op.PrintSizes("Htb=");
        op.decomposer=MatChopDecompFixedTol(0);
//        auto op=HamTBExact(len); op.PrintSizes("HtbExact=");
        DMRG_0_gs sol(op,m,mMax);
        double error=sol.error;
        for(int k=0;k<10;k++)
        {
            std::cout<<"\nsweep "<<k<<"; error="<<error<<"\n\n";
            sol.DoIt_gs();
            sol.DoIt_res();
            std::cout<<"exact="<<ExactEnergyTB(len,len/2,false)<<"\n";
            if (error<1e-12 || error/sol.error<1.1) break;
            error=sol.error;
        }
    }
}

TEST_CASE( "dmrg0 spin1", "[dmrg_0_s1]" )
{
    srand(time(NULL));
    int len=10, m=128, mMax=m;
    std::cout<<std::setprecision(15);

    SECTION( "dmrg" )
    {
        auto op=HamS(3,len,false); op.PrintSizes("Hs1=");
        op.decomposer=MatQRDecomp;
//        op.decomposer=MatChopDecompFixedTol(0);
//        auto sf=SpinFlipGlobal(len);
//        sf.decomposer=MatChopDecompFixedTol(0);
        DMRG_0_gs sol(op,m,mMax,2.0); sol.error=1;
        sol.DoIt_gs();
        for(int k=0;k<8;k++)
        {
            sol.DoIt_resExact();
            std::cout<<"\nsweep "<<k<<"; error="<<sol.error<<"\n\n";
            sol.DoIt_gs();
        }
    }
}

TEST_CASE( "dmrg0 Jacobi-Davidson spin1", "[dmrg_0_jd_s1]" )
{
    srand(time(NULL));
    std::cout<<std::setprecision(15);
    Parameters par;
    par.ReadParameters("param.txt");

    SECTION( "dmrg" )
    {
        auto op=HamS(par.spin,par.length,par.periodic); op.PrintSizes("Hs1=");
        op.decomposer=MatQRDecomp;
//        auto sf=SpinFlipGlobal(len);
//        sf.decomposer=MatChopDecompFixedTol(0);
        DMRG1_JD_gs sol(op,par.m,par.m); sol.error=1;
        sol.DoIt_gs();
        for(int k=0;k<par.nsweep;k++)
        {
            sol.DoIt_res(par.nsweep_resid);
            std::cout<<"\nsweep "<<k<<"; error="<<sol.error<<"\n\n";
            sol.reset_states();
            sol.DoIt_gs();
        }
    }
}

//------------------------- n-step residual correction ----------------------
#include"dmrg_krylov_gs.h"
#include"dmrg_jacobi_davidson_gs.h"

template<class DMRG0>
void TypicalRunDMRG0(MPO op,int nsweep,int m, int nkrylov,int nsweep_resid,
                     int nsite_gs=0, int nsite_resid=1, int nsite_jd=0)
{
    std::cout<<std::setprecision(15);
    op.PrintSizes("Ham=");
    op.decomposer=MatQRDecomp;
    DMRG0 sol(op,m,nkrylov);
    sol.nsite_gs=nsite_gs;
    sol.nsite_resid=nsite_resid;
    sol.nsite_jd=nsite_jd;
    sol.DoIt_gs();
    for(int k=0;k<nsweep;k++)
    {
        sol.DoIt_res(nsweep_resid);
        std::cout<<"\nsweep "<<k+1<<"; error="<<sol.error<<"\n\n";
        sol.reset_states();
        sol.DoIt_gs();
    }
}

TEST_CASE( "dmrg0 Krylov spin1", "[dmrg_0_k_s1]" )
{
    Parameters par("param.txt");
    SECTION( "dmrg" )
    {
        TypicalRunDMRG0<DMRG_krylov_gs>( HamS(par.spin,par.length,par.periodic),
                         par.nsweep,par.m,par.nkrylov,par.nsweep_resid,
                         par.nsite_gs,par.nsite_resid,par.nsweep_jd);
    }
}

TEST_CASE( "dmrg0 JD spin1", "[dmrg_0_jacobi_s1]" )
{
    Parameters par("param.txt");
    SECTION( "dmrg" )
    {
        TypicalRunDMRG0<DMRG_Jacobi_Davidson_gs>( HamS(par.spin,par.length,par.periodic),
                         par.nsweep,par.m,par.nkrylov,par.nsweep_resid,
                         par.nsite_gs,par.nsite_resid,par.nsweep_jd);
    }
}

TEST_CASE( "dmrg0 Krylov spin1/2", "[dmrg_0_k_sf1]" )
{
    Parameters par("param.txt");
    SECTION( "dmrg" )
    {
        TypicalRunDMRG0<DMRG_krylov_gs>( HamSFermi(par.spin,par.length,par.periodic),
                         par.nsweep,par.m,par.nkrylov,par.nsweep_resid,
                         par.nsite_gs,par.nsite_resid,par.nsweep_jd);
    }
}

TEST_CASE( "dmrg0 JD spin1/2", "[dmrg_0_jacobi_sf1]" )
{
    Parameters par("param.txt");
    SECTION( "dmrg" )
    {
        TypicalRunDMRG0<DMRG_Jacobi_Davidson_gs>( HamSFermi(par.spin,par.length,par.periodic),
                         par.nsweep,par.m,par.nkrylov,par.nsweep_resid,
                         par.nsite_gs,par.nsite_resid,par.nsweep_jd);
    }
}


TEST_CASE( "dmrg0 Krylov tigh binding", "[dmrg_0_k_tb]" )
{

    Parameters par("param.txt");
    SECTION( "dmrg" )
    {
        TypicalRunDMRG0<DMRG_krylov_gs>( HamTbAuto(par.length,par.periodic),
                         par.nsweep,par.m,par.nkrylov,par.nsweep_resid,
                         par.nsite_gs,par.nsite_resid,par.nsweep_jd);
        std::cout<<"exact="<<ExactEnergyTB(par.length,par.length/2,par.periodic)<<"\n";
    }
}

TEST_CASE( "dmrg0 JD tigh binding", "[dmrg_0_jacobi_tb]" )
{
    Parameters par("param.txt");
    SECTION( "dmrg" )
    {
        TypicalRunDMRG0<DMRG_Jacobi_Davidson_gs>( HamTbAuto(par.length,par.periodic),
                         par.nsweep,par.m,par.nkrylov,par.nsweep_resid,
                         par.nsite_gs,par.nsite_resid,par.nsweep_jd);
        std::cout<<"exact="<<ExactEnergyTB(par.length,par.length/2,par.periodic)<<"\n";
    }
}

