#include <iostream>
#include"dmrg_res_gs.h"
#include"dmrg1_wse_gs.h"
#include"dmrg_krylov_gs.h"
#include"dmrg_res_cv.h"
#include"tensor.h"
#include"parameters.h"

#include<armadillo>

using namespace std;
using namespace arma;

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

MPO HamTb1(int L,bool periodic)
{
    const int m=20;
    MPSSum h(m,MatSVDFixedTol(1e-13));
    for(int i=0;i<L-1+periodic; i++)
    {
        h += Fermi(i,L,true)*Fermi((i+1)%L,L,false)*(-1.0) ;
        h += Fermi((i+1)%L,L,true)*Fermi(i,L,false)*(-1.0) ;
    }
    return h.toMPS();
}


MPO NParticle(int L)
{
    int m=4;
    MPSSum npart(m,MatSVDFixedTol(1e-13));
    for(int i=0;i<L; i++)
        npart += Fermi(i,L,true)*Fermi(i,L,false) ;
    return npart.toMPS();
}


//---------------------------- Test DMRG basico -------------------------------------------

void TestDMRGBasico(const Parameters &par)
{
    int len=par.length;
    auto op=HamTb1(len,false); op.Sweep(); op.PrintSizes("Hamtb=");
    op.decomposer=MatQRDecomp;
    auto nop=NParticle(len);
    DMRG_krylov_gs sol(op,par.m,par.nkrylov);
    sol.DoIt_gs();
    for(int k=0;k<par.nsweep;k++)
    {
        sol.DoIt_res(par.nsweep_resid);
        std::cout<<"sweep "<<k+1<< "  error="<<sol.error<<" --------------------------------------\n";
        sol.reset_states();
        sol.DoIt_gs();
        cout<<" nT="<<Superblock({&sol.gs[0],&nop,&sol.gs[0]}).value()<<endl;
    }
    cout<<"exact="<<ExactEnergyTB(len,len/2,false);
    ofstream out("gs.dat");
    sol.gs[0].Save(out);
}



int main(int argc, char *argv[])
{
    cout << "Hello World!" << endl;
    std::cout<<std::setprecision(15);
    time_t t0=time(NULL);
    srand(time(NULL));

    if (argc==3 && string(argv[2])=="gs")  // ./a.out parameters.dat gs
    {
        Parameters param;
        param.ReadParameters(argv[1]);
        TestDMRGBasico(param);
    }

    cout<<"\nDone in "<<difftime(time(NULL),t0)<<"s"<<endl;
    return 0;
}
