#include"catch.hpp"
#include"dmrg_0_gs.h"

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
    int len=100, m=16, mMax=128;
    std::cout<<std::setprecision(15);

    SECTION( "dmrg" )
    {
        auto op=HamTbAuto(len,false); op.PrintSizes("Htb=");
//        auto op=HamTBExact(len); op.PrintSizes("HtbExact=");
        DMRG_0_gs sol(op,m,mMax);
        double error=sol.error;
        for(int k=0;k<12;k++)
        {
            std::cout<<"\nsweep "<<k<<"; error="<<error<<"\n\n";
            sol.DoIt();
            std::cout<<"exact="<<ExactEnergyTB(len,len/2,false)<<"\n";
            if (error/sol.error<1.2) break;
            error=sol.error;
        }
    }
}
