#include"catch.hpp"
#include"dmrg_res_gs.h"

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
            sol.DoIt();
            std::cout<<"exact="<<ExactEnergyTB(len,len/2,false)<<"\n";
            if (error<1e-12 || error/sol.error<1.1) break;
            error=sol.error;
        }
    }
}

TEST_CASE( "dmrg0 spin1", "[dmrg_0_s1]" )
{
    srand(time(NULL));
    int len=100, m=200, mMax=m;
    std::cout<<std::setprecision(15);

    SECTION( "dmrg" )
    {
        auto op=HamS1(len,true); op.PrintSizes("Hs1=");
        op.decomposer=MatChopDecompFixedTol(0);
        auto sf=SpinFlipGlobal(len);
        sf.decomposer=MatChopDecompFixedTol(0);
        DMRG_0_gs sol(op,m,mMax);
        double error=sol.error;
        for(int k=0;k<20;k++)
        {
//            std::cout<<"sf="<<sol.sb_sym.value()<<"\n";
            std::cout<<"\nsweep "<<k<<"; error="<<error<<"\n\n";
            sol.DoIt();
//            if (k>2 && error/sol.error<1.1) break;
            error=sol.error;
        }
    }
}
