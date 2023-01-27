#include <iostream>
#include "dmrg1_wse_gs.h"
#include "dmrg1_wse_gs.h"
#include "dmrg_krylov_gs.h"
#include "parameters.h"
#include "ham_irlm.h"

using namespace std;


MPO NParticle(int L)
{
    int m=4;
    MPSSum npart(m,MatSVDFixedTol(1e-13));
    for(int i=0;i<L; i++)
        npart += Fermi(i,L,true)*Fermi(i,L,false) ;
    return npart.toMPS();
}

//---------------------------- Test DMRG basico -------------------------------------------

void TestDMRGBasico(const MPO& ham, const Parameters &par)
{
    int len=ham.length;
    ham.PrintSizes();
    auto nop=NParticle(len);
//    DMRG1_wse_gs sol(ham,par.m,1);
    DMRG_krylov_gs sol(ham,par.m,1);
    sol.tol_diag=1e-12;
    sol.nIterMax=32;
//    sol.gs.decomposer=MatSVDFixedTol(sol.tol_diag);
    for(int k=0;k<3*par.nsweep;k++)
    {
        if (k>=8) sol.nIterMax=64;
        std::cout<<"sweep "<<k+1<<" --------------------------------------\n";
//        for(auto p : MPS::SweepPosSec(len))
//        {
//            sol.SetPos(p);
//            if (k>=par.nsweep) { sol.tol_diag=1e-13; sol.SolveNoWSE(false); }
            sol.DoIt_gs();
            sol.DoIt_res();
            sol.Print();
//        }

        Superblock np({&sol.gs[0],&nop,&sol.gs[0]});
        cout<<" nT="<<np.value()<<endl;
    }

//    sol.gs.Save("gs.dat");
//    sol.gs.decomposer=MatSVDAdaptative(1e-3, par.m);
//    sol.gs.Sweep();
//    sol.gs.PrintSizes("gs=");
}



void CalculateNiN0()
{
    MPS gs;
    gs.Load("gs.dat");
    int N=gs.length;
    arma::mat cicj(N,N);
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++) {
            MPO cc=Fermi(i,N,true)*Fermi(j,N,false);
            cicj(i,j)=Superblock({&gs,&cc,&gs}).value();
        }
    cicj.save("cicj.dat",arma::raw_ascii);
}


int main(int argc, char *argv[])
{
    cout << "Hello World!" << endl;
    std::cout<<std::setprecision(15);
    time_t t0=time(nullptr);
    srand(t0);

    if (argc==4) {
        Parameters param;
        param.ReadParameters("parameters.txt");
        auto ham=HamIRLM(argv[1], argv[2], atof(argv[3]));
        ham.tol=1e-14;
        t0=time(nullptr);
        TestDMRGBasico(ham.Ham(), param);
//        CalculateNiN0();
    }
    else
        cout<<"usage: ./irlm <tfile> <Pfile> U\n";

    cout<<"\nDone in "<<difftime(time(nullptr),t0)<<"s"<<endl;
    return 0;
}
