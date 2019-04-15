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

TEST_CASE( "dmrg0 tight-binding", "[dmrg0_tb]" )
{
    srand(time(NULL));
    int len=10, m=128;
    std::cout<<std::setprecision(15);

    SECTION( "dmrg" )
    {
        auto op=HamTbAuto(len,false); op.PrintSizes("Htb=");
//        auto op=HamTBExact(len); op.PrintSizes("HtbExact=");
        DMRG_0_gs sol(op,m);
        sol.Solve();
        sol.Print();
        for(int k=0;k<4;k++)
        {
            for(auto i:MPS::SweepPosSec(len))
            {
                sol.Solve();
                sol.SetPos(i);
                sol.Print();
            }
            std::cout<<"\n\nsweep\n\n";
            sol.reset_gs();
            std::cout<<"exact="<<ExactEnergyTB(len,len/2,false)<<"\n";
        }
        for(int k=0;k<1;k++)
            for(auto i:MPS::SweepPosSec(len))
            {
//                sol.Solve();
                sol.SetPos(i);
                sol.Print();
            }
        std::cout<<"ener ="<<sol.sb_h11.value()<<"\n";
        std::cout<<"exact="<<ExactEnergyTB(len,len/2,false)<<"\n";
    }
}
