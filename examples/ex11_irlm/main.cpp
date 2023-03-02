#include <iostream>
#include "dmrg.h"
#include "irlm.h"

using namespace std;

int main(int argc, char *argv[])
{
    std::cout<<std::setprecision(15);
    time_t t0=time(nullptr);

    auto model=IRLM{.L=10}.model();
    auto sol=DMRG(model.Ham());
    sol.m=32;
    sol.nIter_diag=32;
    auto Npart=model.NParticle();
    cout<<"sweep energy Npart\n";
    for(auto i=0;i<6;i++) {
        sol.iterate();
        cout<<i+1<<" "<<sol.energy<<" "<<sol.Expectation(Npart)<<endl;
    }
    cout<<"H^2/E^2-1="<<sol.H2(4*sol.m)/pow(sol.energy,2)-1<<endl;

    auto cc=model.CidCj(0,0);
    cout<<"cc="<<sol.correlation(cc,0,0)<<endl;

    cout<<"\nDone in "<<difftime(time(nullptr),t0)<<"s"<<endl;
    return 0;
}
