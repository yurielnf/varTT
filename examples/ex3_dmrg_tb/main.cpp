﻿#include <iostream>
#include"dmrg_res_gs.h"
#include"dmrg1_wse_gs.h"
#include"dmrg_res_cv.h"
#include"tensor.h"
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

MPO Ham2CK1(int L)
{
    int m=1000;
    double lambda=6;
    double V=0.4;
    double U1=3000;
    double U2=1000;
    double Jh=1000;
    double e=-1;
    double D=0;

    MPSSum h(m,MatSVDFixedTol(1e-12));

    //--------------------------------------------------------------------
    //-------------------------impurezas----------------------------------
    //-----------------------------CANALES-ESTRELLA-----------------------
    //--habiamos-hecho-una-transf-unitaria-para-que-quede-forma-estrella----
    //-----------------menos-entanglement-----------------------------------

    mat kin = zeros<mat>(L,L);

    //---------------------------potquimico---------------------------

    kin(0,0)=e;
    kin(L/4,L/4)=e;
    kin(L/2,L/2)=e;
    kin(3*L/4,3*L/4)=e;

    //---------------------------canales------------------------------

    for(int i=1;i<L/4-1;i++)
        for(int j=0;j<4;j++)
        {
            int d=j*L/4;
            kin(i+d,i+1+d)=kin(i+1+d,i+d)=pow((1.0/lambda),(i-1)/2.0);
        }


    //---------------------------hibridizacion------------------------

    kin(0,1)=V;
    kin(1,0)=V;
    kin(L/4,L/4+1)=V;
    kin(L/4+1,L/4)=V;
    kin(L/2,L/2+1)=V;
    kin(L/2+1,L/2)=V;
    kin(3*L/4,3*L/4+1)=V;
    kin(3*L/4+1,3*L/4)=V;

    //----------------------------formaestrella-----------------------

    mat u = zeros<mat>(L,L);

    //----------------------------sitiosimpurezas---------------------

    u(0,0)=1.0;
    u(L/4,L/4)=1.0;
    u(L/2,L/2)=1.0;
    u(3*L/4,3*L/4)=1.0;

    //---------------------------hamiltoniano-banda-------------------

    mat h22 = zeros<mat>(L/4-1,L/4-1);


    for(int i=0;i<L/4-2;i++)
        h22(i,i+1)=h22(i+1,i)=kin(i+1,i+2);

    //--------------------------matriz-que-diagonaliza-h22------------

    vec eigval;
    mat eigvec;

    eig_sym(eigval,eigvec,h22);

    //-----------------------------K-----------------------------------

    for(int i=0; i<L/4-1;i++)
    {
        for(int j=0; j<L/4-1; j++)
        {
            u(i+1,j+1)=u(i+L/4+1,j+L/4+1)=u(i+L/2+1,j+L/2+1)=u(i+3*L/4+1,j+3*L/4+1)=eigvec(i,j);
        }
    }
    mat udaga = u.t();
    mat k;
    k = udaga*kin*u;

    for(int i=0; i<L;i++)
        for(int j=0; j<L;j++)
            if( fabs( k(i,j) ) > 1e-13 )
                h += Fermi(i,L,true)*Fermi(j,L,false)*k(i,j);

    //--------------------------U-same-orb--------------------------------

    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*U1 ;
    h += Fermi(L/2,L,true)*Fermi(L/2,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*U1 ;

    //-------------------------U-dif-orb-----------------------------------

    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*U2 ;
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*U2 ;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*U2 ;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*U2 ;

    //---------------------------'spin-flip'-------------------------------

    h += Fermi(0,L,true)*Fermi(L/2,L,true)*Fermi(0,L,false)*Fermi(L/2,L,false)*Jh ;
    h += Fermi(0,L,true)*Fermi(3*L/4,L,true)*Fermi(L/4,L,false)*Fermi(L/2,L,false)*Jh ;
    h += Fermi(L/4,L,true)*Fermi(L/2,L,true)*Fermi(0,L,false)*Fermi(3*L/4,L,false)*Jh ;

    h += Fermi(L/4,L,true)*Fermi(3*L/4,L,true)*Fermi(L/4,L,false)*Fermi(3*L/4,L,false)*Jh ;
    h += Fermi(0,L,true)*Fermi(L/4,L,true)*Fermi(3*L/4,L,false)*Fermi(L/2,L,false)*Jh ;
    h += Fermi(L/2,L,true)*Fermi(3*L/4,L,true)*Fermi(L/4,L,false)*Fermi(0,L,false)*Jh ;

    //-------------------------------anisotropia----------------------------

    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(0,L,true)*Fermi(0,L,false)*(D/4.0) ;
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*(-D/4.0) ;
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*(D/4.0) ;
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*(-D/4.0) ;

    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(0,L,true)*Fermi(0,L,false)*(-D/4.0) ;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*(D/4.0) ;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*(-D/4.0) ;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*(D/4.0) ;

    h += Fermi(L/2,L,true)*Fermi(L/2,L,false)*Fermi(0,L,true)*Fermi(0,L,false)*(D/4.0) ;
    h += Fermi(L/2,L,true)*Fermi(L/2,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*(-D/4.0) ;
    h += Fermi(L/2,L,true)*Fermi(L/2,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*(D/4.0) ;
    h += Fermi(L/2,L,true)*Fermi(L/2,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*(-D/4.0) ;

    h += Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*Fermi(0,L,true)*Fermi(0,L,false)*(-D/4.0) ;
    h += Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*(D/4.0) ;
    h += Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*(-D/4.0) ;
    h += Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*(D/4.0) ;


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
MPO NImp(int L)
{
    int m=4;
    MPSSum npart(m,MatSVDFixedTol(1e-13));
    for(int i=0;i<4; i++)
        npart += Fermi(i*L/4,L,true)*Fermi(i*L/4,L,false) ;
    return npart.toMPS();
}

void CalculaGS()
{
    int len=10, m=128, mMax=m;

    auto op=HamTbAuto(len,true); op.Sweep(); op.PrintSizes("Hamtb=");
    op.decomposer=MatChopDecompFixedTol(0);
    auto nop=NParticle(len),nimp=NImp(len);
    DMRG_0_gs sol(op,m,mMax,1.0);
    for(int k=0;k<10;k++)
    {
        std::cout<<"sweep "<<k+1<< "  error="<<sol.error<<"\n";
        sol.DoIt_gs();
        Superblock np({&sol.gs,&nop,&sol.gs});
        Superblock ni({&sol.gs,&nimp,&sol.gs});
        cout<<"nImp="<<ni.value()<<" nT="<<np.value()<<endl;
        if (k<9) sol.DoIt_res();
    }
    ofstream out("gs.dat");
    sol.gs.Save(out);
}

void CalculaGreen()
{
    int len=10, m=128;

    auto op=HamTbAuto(len,true); op.Sweep(); op.PrintSizes("Hamtb=");
    op.decomposer=MatChopDecompFixedTol(0);
//    DMRG_0_gs sol(op,m,mMax,1.0);
//    for(int k=0;k<4;k++)
//    {
//        std::cout<<"sweep "<<k+1<< "  error="<<sol.error<<"\n";
//        sol.DoIt_gs();
//        sol.DoIt_res();
//    }

    MPS gs;
    ifstream in("gs.dat");
    gs.Load(in);
    double ener=Superblock({&gs,&op,&gs}).value();

    double w1=-0.1, w2=1, eta=0.1,npoints=50;
    MPS a=Fermi(0,len,false)*gs;
    a.Canonicalize();
    DMRG_0_cv solcv(op,m,a,ener,{w1,w2},{eta,eta});
    std::cout<<"\n\n\nstarting CV\n\n";
    for(int k=0;k<4;k++)
    {
        std::cout<<"sweep "<<k+1<<"\n";
        for(auto p:MPS::SweepPosSec(len))
        {
            solcv.SetPos(p);
            solcv.Solve();
            solcv.Print();
        }
    }
    auto ws=UniformPartition(cmpx(w1,eta),cmpx(w2,eta),npoints);
    auto green=solcv.Green(a,ws);
    ofstream out("green.dat");
    for(uint i=0;i<ws.size();i++)
        out<<ws[i].real()<<" "<<ws[i].imag()<<" "<<green[i].real()<<" "<<green[i].imag()<<"\n";
}

int main()
{
    cout << "Hello World!" << endl;
    std::cout<<std::setprecision(15);
    time_t t0=time(NULL);
    srand(time(NULL));

    CalculaGS();
    CalculaGreen();

    cout<<"\nDone in "<<difftime(time(NULL),t0)<<"s"<<endl;
    return 0;
}
