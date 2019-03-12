#include <iostream>
#include"dmrg_gs.h"

using namespace std;

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

int main()
{
    cout << "Hello World!" << endl;

    srand(time(NULL));
    int len=20, m=128;
    MPS x(len,m);
    x.FillRandu({m,2,m});
    x.Normalize();
    x.PrintSizes("|x>=");
    auto op=HamTB2(len,false);
    op.PrintSizes("H=");
    DMRG_gs sol(op,m);
    sol.Solve();
//    for(int k=0;k<4;k++)
//        for(int i:MPS::SweepPosSec(len))
//        {
//            sol.Solve();
//            sol.sb.SetPos(i);
//        }
    std::cout<<"exact ener="<<ExactEnergyTB(len,len/2,false)<<"\n";
    return 0;
}
