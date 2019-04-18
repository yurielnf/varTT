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
//    Index dim={4,4,1,2,2};
//    Index id(dim.size(),0);
//    int pos=0;
//    for(int i=0;i<Prod(dim);i++)
//    {
//        for(uint j=0;j<dim.size();j++)
//            cout<<id[j]<<" ";
//        cout<<"\n";
//        id[pos]++;
//        if (id[pos]==dim[pos])
//        {
//            while (id[pos]==dim[pos]) {pos++; id[pos]++;}
//            for(int j=0;j<pos;j++) id[j]=0;
//            pos=0;
//        }
//    }
//    return 0;

    cout << "Hello World!" << endl;
    time_t t0=time(NULL);
    srand(time(NULL));
    int len=100, m=128;

    //SECTION( "dmrg" )
    {
        auto op=HamTbAuto(len,false);
//        auto op=HamTBExact(len);
        DMRG_gs sol(op,m);
//        sol.nIterMax=20;
        sol.tol_diag=1e-5;
        sol.Solve();
        for(int k=0;k<2;k++)
        for(auto i:MPS::SweepPosSec(len))
        {
            sol.Solve();
            sol.sb.SetPos(i);
            sol.Print();
        }
        std::cout<<"ener ="<<sol.sb.value()<<"\n";
        std::cout<<"exact ener="<<ExactEnergyTB(len,len/2,false)<<"\n";
    }

    cout<<"\nDone in "<<difftime(time(NULL),t0)<<"s"<<endl;
    return 0;
}
