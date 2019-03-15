#include"catch.hpp"
#include"dmrg_gs.h"

double ExactEnergyTB(int L, int nPart,bool periodic)
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
    int len=20, m=128;

    SECTION( "dmrg" )
    {
        auto op=HamTB2(len,false);
//        auto op=HamTBExact(len);
        DMRG_gs sol(op,m);
        sol.Solve();
        for(int k=0;k<2;k++)
        for(int i:MPS::SweepPosSec(len))
        {
            sol.Solve();
            sol.sb.SetPos(i);
            sol.Print();
        }
        std::cout<<"exact ener="<<ExactEnergyTB(len,len/2,false)<<"\n";
    }
}
