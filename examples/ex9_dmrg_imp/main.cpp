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


MPO HamSIAMTBStar(int L)
{
    int m=2;
    double U=4;
    double epsilon=-2;
    //---interaccion-QD-bandas
    double V=0.6;
    double V1=1.5*V;


    MPSSum h(m,MatSVDFixedTol(1e-12));

    //-------------------------impurezas----------------------------------
    //-----------------------CANALES-ESTRELLA-----------------------------

    mat kin = zeros<mat>(L,L);

    //-------------ENERGIAS-DIAGONALES-NIVELES-QD---------------------

    kin(0,0)=epsilon;
    kin(L/2,L/2)=epsilon;

    //-----------------------canales------------------------------
    //----------------------hoppings -----------------------------

    for(int i=1;i<L/2-1;i++)
        for(int j=0;j<2;j++)
        {
            int d=j*L/2;
            kin(i+d,i+1+d)=kin(i+1+d,i+d)=1.0;
        }

    //---------------------------hibridizacion------------------------

    kin(0,1)=kin(1,0)=V1;
    kin(L/2,L/2+1)=kin(L/2+1,L/2)=V1;

    //----------------------------formaestrella-----------------------

    mat u = zeros<mat>(L,L);

    //----------------------------sitiosimpurezas---------------------

    u(0,0)=1.0;
    u(L/2,L/2)=1.0;

    //---------------------------hamiltoniano-banda-------------------

    mat h22 = zeros<mat>(L/2-1,L/2-1);
    for(int i=0;i<L/2-2;i++)
        h22(i,i+1)=h22(i+1,i)=kin(i+1,i+2);

    //--------------------------matriz-que-diagonaliza-h22------------

    vec eigval;
    mat eigvec;

    eig_sym(eigval,eigvec,h22);

    //-----------------------------K-----------------------------------

    for(int i=0; i<L/2-1;i++)
    {
        for(int j=0; j<L/2-1; j++)
        {
            u(i+1,j+1)=u(i+L/2+1,j+L/2+1)=eigvec(i,j);
        }
    }
    mat udaga = u.t();
    mat k;
    k = udaga*kin*u;

    for(int i=0; i<L;i++)
        for(int j=0; j<L;j++)
            if( fabs( k(i,j) ) > 1e-13 )
                h += Fermi(i,L,true)*Fermi(j,L,false)*k(i,j);


    //-----------------Termino---U--------------------------------------

    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*U;

    return h.toMPS().Sweep();
}


MPO HamSIAMNRGStar(int L, double BF)
{
    int m=2;
    double lambda=2;
    double U=4;
    double B=BF;
    double epsilon=-2;
    //---interaccion-QD-bandas
    double V=0.6;
    double V2=-V;


    MPSSum h(m,MatSVDFixedTol(1e-12));

    //--------------------------------------------------------------------
    //-------------------------impurezas----------------------------------
    //-----------------------CANALES-ESTRELLA-----------------------------

    mat kin = zeros<mat>(L,L);

    //-------------ENERGIAS-DIAGONALES-NIVELES-QD---------------------

    kin(0,0)=epsilon-0.5*B;
    kin(L/2,L/2)=epsilon+0.5*B;


    //---------------------------canales------------------------------
    //-----------------------hoppings NRG-----------------------------

    for(int i=1;i<L/2-1;i++)
        for(int j=0;j<2;j++)
        {
            int d=j*L/2;
            kin(i+d,i+1+d)=kin(i+1+d,i+d)=pow((1.0/lambda),(i-1)/2.0);
        }

    //---------------------------hibridizacion------------------------

    kin(0,1)=V2;
    kin(1,0)=V2;
    kin(L/2,L/2+1)=V2;
    kin(L/2+1,L/2)=V2;

    //----------------------------formaestrella-----------------------

    mat u = zeros<mat>(L,L);

    //----------------------------sitiosimpurezas---------------------

    u(0,0)=1.0;
    u(L/2,L/2)=1.0;

    //---------------------------hamiltoniano-banda-------------------

    mat h22 = zeros<mat>(L/2-1,L/2-1);
    for(int i=0;i<L/2-2;i++)
        h22(i,i+1)=h22(i+1,i)=kin(i+1,i+2);

    //--------------------------matriz-que-diagonaliza-h22------------

    vec eigval;
    mat eigvec;

    eig_sym(eigval,eigvec,h22);

    //-----------------------------K-----------------------------------

    for(int i=0; i<L/2-1;i++)
    {
        for(int j=0; j<L/2-1; j++)
        {
            u(i+1,j+1)=u(i+L/2+1,j+L/2+1)=eigvec(i,j);
        }
    }
    mat udaga = u.t();
    mat k;
    k = udaga*kin*u;

    for(int i=0; i<L;i++)
        for(int j=0; j<L;j++)
            if( fabs( k(i,j) ) > 1e-13 )
                h += Fermi(i,L,true)*Fermi(j,L,false)*k(i,j);


    //-----------------Termino---U--------------------------------------

    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*U;



    return h.toMPS().Sweep();
}




MPO NParticle(int L)
{
    int m=4;
    MPSSum npart(m,MatSVDFixedTol(1e-13));
    for(int i=0;i<L; i++)
        npart += Fermi(i,L,true)*Fermi(i,L,false) ;
    return npart.toMPS();
}

void CalculateNi(const Parameters par)
{
    MPS gs;
    gs.Load("gs.dat");
    ofstream out("ni.dat");
    int L=par.length;
    for(int i=0; i<L; i++)
    {
        MPO rr=Fermi(i,L,true)*Fermi(i,L,false);
        double ni=Superblock({&gs,&rr,&gs}).value();
        out<<i+1<<" "<<ni<<endl;
    }
}


//---------------------------- Test DMRG basico -------------------------------------------

void TestDMRGBasico(const Parameters &par)
{
    int len=par.length;
    auto op=HamSIAMTBStar(len); op.Sweep(); op.PrintSizes("HamSiamMPO=");
    op.decomposer=MatQRDecomp;
    auto nop=NParticle(len);

    DMRG1_wse_gs sol(op,par.m);
    sol.tol_diag=1e-4;
    for(int k=0;k<par.nsweep;k++)
    {
        std::cout<<"sweep "<<k+1<<" --------------------------------------\n";
        if (k==par.nsweep-1) sol.tol_diag=1e-7;
        for(auto p : MPS::SweepPosSec(op.length))
        {
            sol.SetPos(p);
            sol.Solve();
            if ((p.i+1) % (op.length/10) ==0) sol.Print();
        }
        cout<<" nT="<<Superblock({&sol.gs,&nop,&sol.gs}).value()<<endl;
    }
    ofstream out("gs.dat");
    sol.gs.Save(out);
}


int main(int argc, char *argv[])
{
    cout << "Hello World!" << endl;
    std::cout<<std::setprecision(15);
    time_t t0=time(NULL);
    srand(time(NULL));

    if (argc==3 && string(argv[2])=="basic")  // ./a.out parameters.dat basic
    {
        Parameters param;
        param.ReadParameters(argv[1]);
        TestDMRGBasico(param);
        CalculateNi(param);
    }
    cout<<"\nDone in "<<difftime(time(NULL),t0)<<"s"<<endl;
    return 0;
}
