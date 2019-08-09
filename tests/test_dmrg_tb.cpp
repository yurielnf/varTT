#include"catch.hpp"
#include"parameters.h"
#include"dmrg_gs.h"
#include"dmrg_se_gs.h"
#include"dmrg_wse_gs.h"

#include"dmrg1_gs.h"
#include"dmrg1_wse_gs.h"

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

TEST_CASE( "dmrg tight-binding", "[dmrg_tb]" )
{
    srand(time(NULL));
    int len=10, m=128;
    std::cout<<std::setprecision(15);
    SECTION( "dmrg" )
    {
        auto op=HamTbAuto(len,false); op.PrintSizes("Htb=");
        op.decomposer=MatChopDecompFixedTol(0);
//        auto op=HamTBExact(len); op.PrintSizes("HtbExact=");
        DMRG1_gs sol(op,m);
        for(int k=0;k<2;k++)
        {
            for(auto p:MPS::SweepPosSec(len))
            {
                sol.SetPos(p);
                sol.Solve();
                if ((p.i+1) % (len/10) ==0) sol.Print();
            }
            std::cout<<"ener ="<<sol.sb.value()<<"\n";
        }
        std::cout<<"exact="<<ExactEnergyTB(len,len/2,false)<<"\n";
    }
}

TEST_CASE( "dmrg S=1", "[dmrg_s1]" )
{
    srand(time(NULL));
    int len=10, m=128;
    std::cout<<std::setprecision(15);
    SECTION( "dmrg" )
    {
        auto op=HamS1(len,true); op.PrintSizes("Hs1=");
        op.decomposer=MatChopDecompFixedTol(0);
        auto sf=SpinFlipGlobal(len); sf.decomposer=MatChopDecompFixedTol(0);
//        op+=op*sf*(1.0);
//        op+=sf*op*(1.0);
//        op+=sf*op*sf;
//        op*=0.5;
//        op.decomposer=MatChopDecompFixedTol(0);
        DMRG_gs sol(op,m);
        sol.tol_diag=1e-3;
        for(int k=0;k<10;k++)
        {
            for(auto p:MPS::SweepPosSec(len))
            {
                sol.SetPos(p);
                sol.Solve();
                if ((p.i+1) % (len/10) ==0) sol.Print();
            }
            if (k>=3) {sol.tol_diag=1e-9; }
            if (k>=8) {sol.tol_diag=1e-11;}

//            std::cout<<"sf="<<sol.sb_sym.value()<<"\n";
//            sol.Reset_gs();
        }
    }
}

TEST_CASE( "dmrg with White subspace-expansion tight-binding", "[dmrg_wse_tb]" )
{
    srand(time(NULL));
    int len=10, m=128;
    std::cout<<std::setprecision(15);
    SECTION( "dmrg" )
    {
        auto op=HamTbAuto(len,false); op.PrintSizes("Htb=");
        op.decomposer=MatChopDecompFixedTol(0);
//        auto op=HamTBExact(len); op.PrintSizes("HtbExact=");
        DMRG1_wse_gs sol(op,m);
        for(int k=0;k<10;k++)
        {
            for(auto p:MPS::SweepPosSec(len))
            {
                sol.SetPos(p);
                sol.Solve();
                if ((p.i+1) % (len/10) ==0) sol.Print();
            }
            std::cout<<"exact="<<ExactEnergyTB(len,len/2,false)<<"\n";
        }
    }
}


TEST_CASE( "dmrg with White subspace-expansion S=1", "[dmrg_wse_s1]" )
{
    srand(time(NULL));
    Parameters par;
    par.ReadParameters("param.txt");
    std::cout<<std::setprecision(15);
    SECTION( "dmrg" )
    {
        auto op=HamS(par.spin, par.length, par.periodic); op.PrintSizes("Hs1=");
        op.decomposer=MatChopDecompFixedTol(0);
//        auto sf=SpinFlipGlobal(len); sf.decomposer=MatChopDecompFixedTol(0);
        DMRG1_wse_gs sol(op,par.m);
        sol.tol_diag=1e-6;
        for(int k=0;k<par.nsweep;k++)
        {
            for(auto p : MPS::SweepPosSec(par.length))
            {
                sol.SetPos(p);
                sol.Solve();
                if ((p.i+1) % (par.length/10) ==0) sol.Print();
            }
            std::cout<<"sweep "<<k+1<<"\n";
//            std::cout<<"sf="<<sol.sb_sym.value()<<"\n";
//            sol.Reset_gs();
            if (k>=3) {sol.tol_diag=1e-9; }
            if (k>=8) {sol.tol_diag=1e-11; }
            if (k>=12){sol.tol_diag=1e-13; }
//            if (k>=16){sol.alpha=0;}
        }
    }
}
