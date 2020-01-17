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

MPO Ham2CK1(int L,double DSz2)
{
    int m=1000;
    double lambda=6;
    double V=1.13;
    double U1=30;
    double U2=10;
    double Jh=10;
    double e=-2;
    double D=DSz2;

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

MPO HamICTP1(int L, double BF)
{
    int m=2;
    double lambda=6;
    double Ec=5;
    double J=1.0;
    double B=BF;
    double epsilon=-15.75;
    //---interaccion-QD-bandas
    double V=0.9;
    double V1=1.5*V;
    double V2=-V;

    MPSSum h(m,MatSVDFixedTol(1e-12));

    //--------------------------------------------------------------------
    //-------------------------impurezas----------------------------------
    //-----------------------CANALES-ESTRELLA-----------------------------

    mat kin = zeros<mat>(L,L);

    //--------------------energia-sitios-impurezas--------------------
    //------------LVL1UP--BAÑO-UP---LVL1DOWN--BAÑO-DOWN---------------
    //------------LVL2UP--BAÑO-UP---LVL2DOWN--BAÑO-DOWN---------------

    //-------------ENERGIAS-DIAGONALES-NIVELES-QD---------------------

    kin(0,0)=kin(L/2,L/2)=epsilon-0.5*B;
    kin(L/4,L/4)=kin(3*L/4,3*L/4)=epsilon+0.5*B;


    //---------------------------canales------------------------------
    //-----------------------hoppings NRG-----------------------------

    for(int i=1;i<L/4-1;i++)
        for(int j=0;j<4;j++)
        {
            int d=j*L/4;
            kin(i+d,i+1+d)=kin(i+1+d,i+d)=pow((1.0/lambda),(i-1)/2.0);
        }

    //---------------------------hibridizacion------------------------

    kin(0,1)=V1;
    kin(1,0)=V1;
    kin(L/4,L/4+1)=V1;
    kin(L/4+1,L/4)=V1;
    kin(L/2,L/2+1)=V2;
    kin(L/2+1,L/2)=V2;
    kin(3*L/4,3*L/4+1)=V2;
    kin(3*L/4+1,3*L/4)=V2;

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

    //-----------------Termino---Ec*N^2---------------------------------

    for(int i=0;i<4;i++)
    {
        int d=i*L/4;
        for(int j=0;j<4;j++)
        {
            int q=j*L/4;
            h += Fermi(d,L,true)*Fermi(d,L,false)*Fermi(q,L,true)*Fermi(q,L,false)*Ec;
        }
    }

    //-------------Termino------J*S^2----------------------------------------
    //------------Sx^2+Sy^2------------------------------------------------
    h += Fermi(0,L,true)*Fermi(L/4,L,false)*Fermi(L/4,L,true)*Fermi(0,L,false)*(-0.5)*J;
    h += Fermi(0,L,true)*Fermi(L/4,L,false)*Fermi(3*L/4,L,true)*Fermi(L/2,L,false)*(-0.5)*J;
    h += Fermi(L/4,L,true)*Fermi(0,L,false)*Fermi(0,L,true)*Fermi(L/4,L,false)*(-0.5)*J;
    h += Fermi(L/4,L,true)*Fermi(0,L,false)*Fermi(L/2,L,true)*Fermi(3*L/4,L,false)*(-0.5)*J;
    h += Fermi(L/2,L,true)*Fermi(3*L/4,L,false)*Fermi(L/4,L,true)*Fermi(0,L,false)*(-0.5)*J;
    h += Fermi(L/2,L,true)*Fermi(3*L/4,L,false)*Fermi(3*L/4,L,true)*Fermi(L/2,L,false)*(-0.5)*J;
    h += Fermi(3*L/4,L,true)*Fermi(L/2,L,false)*Fermi(0,L,true)*Fermi(L/4,L,false)*(-0.5)*J;
    h += Fermi(3*L/4,L,true)*Fermi(L/2,L,false)*Fermi(L/2,L,true)*Fermi(3*L/4,L,false)*(-0.5)*J;

    //---------Sz^2-------------------------------------------------------
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(0,L,true)*Fermi(0,L,false)*(-0.25)*J;
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*0.25*J;
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*(-0.25)*J;
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*0.25*J;

    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(0,L,true)*Fermi(0,L,false)*0.25*J;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*(-0.25)*J;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*0.25*J;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*(-0.25)*J;

    h += Fermi(L/2,L,true)*Fermi(L/2,L,false)*Fermi(0,L,true)*Fermi(0,L,false)*(-0.25)*J;
    h += Fermi(L/2,L,true)*Fermi(L/2,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*0.25*J;
    h += Fermi(L/2,L,true)*Fermi(L/2,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*(-0.25)*J;
    h += Fermi(L/2,L,true)*Fermi(L/2,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*0.25*J;

    h += Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*Fermi(0,L,true)*Fermi(0,L,false)*0.25*J;
    h += Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*(-0.25)*J;
    h += Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*0.25*J;
    h += Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*(-0.25)*J;


    return h.toMPS().Sweep();
}


MPO HamICTP2(int L, double BF)
{
    int m=2;
    double lambda=6;
    double Ec=5;
    double J=1.0;
    double B=BF;
    double epsilon=-15.75;
    //---interaccion-QD-bandas
    double V=0.9;
    double V1=1.5*V;
    double V2=-V;

    MPSSum h(m,MatSVDFixedTol(1e-12));

    //--------------------------------------------------------------------
    //-------------------------impurezas----------------------------------
    //-----------------------CANALES-ESTRELLA-----------------------------

    mat kin = zeros<mat>(L,L);

    //--------------------energia-sitios-impurezas--------------------
    //------------LVL1UP--BAÑO-UP---LVL2UP--BAÑO-UP---------------
    //------------BAÑO-DW--LVL2DW---BAÑO-DW--LVL1DW-------------

    //-------------ENERGIAS-DIAGONALES-NIVELES-QD---------------------

    kin(0,0)=kin(L/4,L/4)=epsilon-0.5*B;
    kin(L/2,L/2)=kin(3*L/4,3*L/4)=epsilon+0.5*B;

    //---------------------------canales------------------------------
    //-----------------------hoppings NRG-----------------------------

    for(int i=1;i<L/4-1;i++)
        for(int j=0;j<2;j++)
        {
            int d=j*L/4;
            kin(i+d,i+1+d)=kin(i+1+d,i+d)=pow((1.0/lambda),(i-1)/2.0);
        }
    for(int i=3*L/4-2;i>L/2;i--)
        for(int j=0;j<2;j++)
        {
            int d=j*L/4;
            kin(i+d,i-1+d)=kin(i-1+d,i+d)=pow((1.0/lambda),(3*L/4-2-i)/2.0);
        }

    //---------------------------hibridizacion------------------------

    kin(0,1)=kin(1,0)=V1;
    kin(L/4,L/4+1)=kin(L/4+1,L/4)=V2;
    kin(3*L/4-1,3*L/4-2)=kin(3*L/4-2,3*L/4-1)=V2;
    kin(L-1,L-2)=kin(L-2,L-1)=V1;

    //----------------------------formaestrella-----------------------

    mat u = zeros<mat>(L,L);

    //----------------------------sitiosimpurezas---------------------

    u(0,0)=1.0;
    u(L/4,L/4)=1.0;
    u(3*L/4-1,3*L/4-1)=1.0;
    u(L-1,L-1)=1.0;

    //-------------hamiltoniano-banda-primera-mitad---------------------

    mat h22 = zeros<mat>(L/4-1,L/4-1);
    for(int i=0;i<L/4-2;i++)
        h22(i,i+1)=h22(i+1,i)=kin(i+1,i+2);

    //-----------matriz-que-diagonaliza-h22------------------------

    vec eigval;
    mat eigvec;
    eig_sym(eigval,eigvec,h22);

    for(int i=0; i<L/4-1;i++)
        for(int j=0; j<L/4-1; j++)
            u(i+1,j+1)=u(i+L/4+1,j+L/4+1)=eigvec(i,j);

    //-------------hamiltoniano-banda-segunda-mitad------------------

    mat b22 = zeros<mat>(L/4-1,L/4-1);
    for(int i=0;i<L/4-2;i++)
        b22(i,i+1)=b22(i+1,i)=kin(i+L/2,i+L/2+1);

    //-------------matriz-que-diagonaliza-b22------------------------

    vec eigvalb;
    mat eigvecb;
    eig_sym(eigvalb,eigvecb,b22);

    for(int i=0; i<L/4-1;i++)
        for(int j=0; j<L/4-1; j++)
            u(i+L/2+1,j+L/2+1)=u(i+3*L/4+1,j+3*L/4+1)=eigvecb(i,j);

    //--------------------K------------------------------------------

    mat udaga = u.t();
    mat k;
    k = udaga*kin*u;

    for(int i=0; i<L;i++)
        for(int j=0; j<L;j++)
            if( fabs( k(i,j) ) > 1e-13 )
                h += Fermi(i,L,true)*Fermi(j,L,false)*k(i,j);

    //-----------------Termino---Ec*N^2---------------------------------

    for(int i=0;i<2;i++)
    {
        int d=i*L/4;
        for(int j=0;j<2;j++)
        {
            int q=j*L/4;
            h += Fermi(d,L,true)*Fermi(d,L,false)*Fermi(q,L,true)*Fermi(q,L,false)*Ec;
            h += Fermi(d,L,true)*Fermi(d,L,false)*Fermi(q+3*L/4-1,L,true)*Fermi(q+3*L/4-1,L,false)*Ec;
            h += Fermi(d+3*L/4-1,L,true)*Fermi(d+3*L/4-1,L,false)*Fermi(q,L,true)*Fermi(q,L,false)*Ec;
            h += Fermi(d+3*L/4-1,L,true)*Fermi(d+3*L/4-1,L,false)*Fermi(q+3*L/4-1,L,true)*Fermi(q+3*L/4-1,L,false)*Ec;
        }
    }

    //-------------Termino------J*S^2----------------------------------------
    //------------Sx^2+Sy^2------------------------------------------------
    h += Fermi(0,L,true)*Fermi(L-1,L,false)*Fermi(L-1,L,true)*Fermi(0,L,false)*(-0.5)*J;
    h += Fermi(0,L,true)*Fermi(L-1,L,false)*Fermi(3*L/4-1,L,true)*Fermi(L/4,L,false)*(-0.5)*J;
    h += Fermi(L-1,L,true)*Fermi(0,L,false)*Fermi(0,L,true)*Fermi(L-1,L,false)*(-0.5)*J;
    h += Fermi(L-1,L,true)*Fermi(0,L,false)*Fermi(L/4,L,true)*Fermi(3*L/4-1,L,false)*(-0.5)*J;
    h += Fermi(L/4,L,true)*Fermi(3*L/4-1,L,false)*Fermi(L-1,L,true)*Fermi(0,L,false)*(-0.5)*J;
    h += Fermi(L/4,L,true)*Fermi(3*L/4-1,L,false)*Fermi(3*L/4-1,L,true)*Fermi(L/4,L,false)*(-0.5)*J;
    h += Fermi(3*L/4-1,L,true)*Fermi(L/4,L,false)*Fermi(0,L,true)*Fermi(L-1,L,false)*(-0.5)*J;
    h += Fermi(3*L/4-1,L,true)*Fermi(L/4,L,false)*Fermi(L/4,L,true)*Fermi(3*L/4-1,L,false)*(-0.5)*J;

    //---------Sz^2-------------------------------------------------------
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(0,L,true)*Fermi(0,L,false)*(-0.25)*J;
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(L-1,L,true)*Fermi(L-1,L,false)*0.25*J;
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*(-0.25)*J;
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(3*L/4-1,L,true)*Fermi(3*L/4-1,L,false)*0.25*J;

    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(0,L,true)*Fermi(0,L,false)*(-0.25)*J;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(L-1,L,true)*Fermi(L-1,L,false)*0.25*J;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*(-0.25)*J;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(3*L/4-1,L,true)*Fermi(3*L/4-1,L,false)*0.25*J;

    h += Fermi(3*L/4-1,L,true)*Fermi(3*L/4-1,L,false)*Fermi(0,L,true)*Fermi(0,L,false)*0.25*J;
    h += Fermi(3*L/4-1,L,true)*Fermi(3*L/4-1,L,false)*Fermi(L-1,L,true)*Fermi(L-1,L,false)*(-0.25)*J;
    h += Fermi(3*L/4-1,L,true)*Fermi(3*L/4-1,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*0.25*J;
    h += Fermi(3*L/4-1,L,true)*Fermi(3*L/4-1,L,false)*Fermi(3*L/4-1,L,true)*Fermi(3*L/4-1,L,false)*(-0.25)*J;

    h += Fermi(L-1,L,true)*Fermi(L-1,L,false)*Fermi(0,L,true)*Fermi(0,L,false)*0.25*J;
    h += Fermi(L-1,L,true)*Fermi(L-1,L,false)*Fermi(L-1,L,true)*Fermi(L-1,L,false)*(-0.25)*J;
    h += Fermi(L-1,L,true)*Fermi(L-1,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*0.25*J;
    h += Fermi(L-1,L,true)*Fermi(L-1,L,false)*Fermi(3*L/4-1,L,true)*Fermi(3*L/4-1,L,false)*(-0.25)*J;


    return h.toMPS().Sweep();
}

struct Reverse
{
    int L;
    int operator()(int i) const {return L-1-i;}
};

MPO HamICTP3(int L, double BF)
{
    int m=2;
    double lambda=6;
    double Ec=5;
    double J=1.0;
    double B=BF;
    double epsilon=-15.75;
    //---interaccion-QD-bandas
    double V=0.9;
    double V1=1.5*V;
    double V2=-V;

    MPSSum h(m,MatSVDFixedTol(1e-12));
    Reverse R{L};
    vector<int> p={0,R(0),L/4,R(L/4)};

    //--------------------------------------------------------------------
    //-------------------------impurezas----------------------------------
    //-----------------------CANALES-ESTRELLA-----------------------------

    mat kin = zeros<mat>(L,L);

    //--------------------energia-sitios-impurezas--------------------
    //------------LVL1UP--BAÑO-UP---LVL1DOWN--BAÑO-DOWN---------------
    //------------LVL2UP--BAÑO-UP---LVL2DOWN--BAÑO-DOWN---------------

    //-------------ENERGIAS-DIAGONALES-NIVELES-QD---------------------

    kin(0,0)=kin(L/4,L/4)=epsilon-0.5*B;
    kin(R(0),R(0))=kin(R(L/4),R(L/4))=epsilon+0.5*B;


    //---------------------------canales------------------------------
    //-----------------------hoppings NRG-----------------------------

    for(int i=1;i<L/4-1;i++)
        for(int j=0;j<2;j++)
        {
            int d=j*L/4;
            kin(i+d,i+1+d)=
                    kin(i+1+d,i+d)=
                    kin(R(i+d),R(i+1+d))=
                    kin(R(i+1+d),R(i+d))=pow((1.0/lambda),(i-1)/2.0);
        }

    //---------------------------hibridizacion------------------------

    kin(0,1)=
            kin(1,0)=
            kin(R(0),R(1))=
            kin(R(1),R(0))=V1;
    kin(L/4,L/4+1)=
            kin(L/4+1,L/4)=
            kin(R(L/4),R(L/4+1))=
            kin(R(L/4+1),R(L/4))=V2;

    //----------------------------formaestrella-----------------------

    mat u = zeros<mat>(L,L);

    //----------------------------sitiosimpurezas---------------------

    for(int i:p) u(i,i)=1;

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
            u(i+1,j+1)=u(i+L/4+1,j+L/4+1)=
                    u(R(i+1),R(j+1))=u(R(i+L/4+1),R(j+L/4+1))=eigvec(i,j);
        }
    }
    mat udaga = u.t();
    mat k;
    k = udaga*kin*u;

    for(int i=0; i<L;i++)
        for(int j=0; j<L;j++)
            if( fabs( k(i,j) ) > 1e-13 )
                h += Fermi(i,L,true)*Fermi(j,L,false)*k(i,j);

    //-----------------Termino---Ec*N^2---------------------------------


    for(int i:p)
        for(int j:p)
            h += Fermi(i,L,true)*Fermi(i,L,false)*Fermi(j,L,true)*Fermi(j,L,false)*Ec;

    //-------------Termino------J*S^2----------------------------------------
    //------------Sx^2+Sy^2------------------------------------------------
    h += Fermi(p[0],L,true)*Fermi(p[1],L,false)*Fermi(p[1],L,true)*Fermi(p[0],L,false)*(-0.5)*J;
    h += Fermi(p[0],L,true)*Fermi(p[1],L,false)*Fermi(p[3],L,true)*Fermi(p[2],L,false)*(-0.5)*J;
    h += Fermi(p[1],L,true)*Fermi(p[0],L,false)*Fermi(p[0],L,true)*Fermi(p[1],L,false)*(-0.5)*J;
    h += Fermi(p[1],L,true)*Fermi(p[0],L,false)*Fermi(p[2],L,true)*Fermi(p[3],L,false)*(-0.5)*J;
    h += Fermi(p[2],L,true)*Fermi(p[3],L,false)*Fermi(p[1],L,true)*Fermi(p[0],L,false)*(-0.5)*J;
    h += Fermi(p[2],L,true)*Fermi(p[3],L,false)*Fermi(p[3],L,true)*Fermi(p[2],L,false)*(-0.5)*J;
    h += Fermi(p[3],L,true)*Fermi(p[2],L,false)*Fermi(p[0],L,true)*Fermi(p[1],L,false)*(-0.5)*J;
    h += Fermi(p[3],L,true)*Fermi(p[2],L,false)*Fermi(p[2],L,true)*Fermi(p[3],L,false)*(-0.5)*J;

    //---------Sz^2-------------------------------------------------------
    h += Fermi(p[0],L,true)*Fermi(p[0],L,false)*Fermi(p[0],L,true)*Fermi(p[0],L,false)*(-0.25)*J;
    h += Fermi(p[0],L,true)*Fermi(p[0],L,false)*Fermi(p[1],L,true)*Fermi(p[1],L,false)*0.25*J;
    h += Fermi(p[0],L,true)*Fermi(p[0],L,false)*Fermi(p[2],L,true)*Fermi(p[2],L,false)*(-0.25)*J;
    h += Fermi(p[0],L,true)*Fermi(p[0],L,false)*Fermi(p[3],L,true)*Fermi(p[3],L,false)*0.25*J;

    h += Fermi(p[1],L,true)*Fermi(p[1],L,false)*Fermi(p[0],L,true)*Fermi(p[0],L,false)*0.25*J;
    h += Fermi(p[1],L,true)*Fermi(p[1],L,false)*Fermi(p[1],L,true)*Fermi(p[1],L,false)*(-0.25)*J;
    h += Fermi(p[1],L,true)*Fermi(p[1],L,false)*Fermi(p[2],L,true)*Fermi(p[2],L,false)*0.25*J;
    h += Fermi(p[1],L,true)*Fermi(p[1],L,false)*Fermi(p[3],L,true)*Fermi(p[3],L,false)*(-0.25)*J;

    h += Fermi(p[2],L,true)*Fermi(p[2],L,false)*Fermi(p[0],L,true)*Fermi(p[0],L,false)*(-0.25)*J;
    h += Fermi(p[2],L,true)*Fermi(p[2],L,false)*Fermi(p[1],L,true)*Fermi(p[1],L,false)*0.25*J;
    h += Fermi(p[2],L,true)*Fermi(p[2],L,false)*Fermi(p[2],L,true)*Fermi(p[2],L,false)*(-0.25)*J;
    h += Fermi(p[2],L,true)*Fermi(p[2],L,false)*Fermi(p[3],L,true)*Fermi(p[3],L,false)*0.25*J;

    h += Fermi(p[3],L,true)*Fermi(p[3],L,false)*Fermi(p[0],L,true)*Fermi(p[0],L,false)*0.25*J;
    h += Fermi(p[3],L,true)*Fermi(p[3],L,false)*Fermi(p[1],L,true)*Fermi(p[1],L,false)*(-0.25)*J;
    h += Fermi(p[3],L,true)*Fermi(p[3],L,false)*Fermi(p[2],L,true)*Fermi(p[2],L,false)*0.25*J;
    h += Fermi(p[3],L,true)*Fermi(p[3],L,false)*Fermi(p[3],L,true)*Fermi(p[3],L,false)*(-0.25)*J;


    return h.toMPS().Sweep();
}

MPO Ham2SKtoy(int L, double BF, double Jh)
{
    int m=2;
    double lambda=6;
    double B=BF;
    double J=Jh;
    //---interaccion-QD-bandas
    double Tk1=1;
    double Tk2=10;

    MPSSum h(m,MatSVDFixedTol(1e-12));

    //-------------------parte kin------------------------------------

    mat kin = zeros<mat>(L,L);

    //-----------------------FORMA------------------------------------
    //------------LVL1UP--BAÑO-UP---LVL1DOWN--BAÑO-DOWN---------------
    //------------LVL2UP--BAÑO-UP---LVL2DOWN--BAÑO-DOWN---------------
    //--------------------energia-sitios-impurezas--------------------

    kin(0,0)=kin(L/2,L/2)=-0.5*B;
    kin(L/4,L/4)=kin(3*L/4,3*L/4)=0.5*B;

    //---------------------------canales------------------------------
    //-----------------------hoppings NRG-----------------------------

    for(int i=1;i<L/4-1;i++)
        for(int j=0;j<4;j++)
        {
            int d=j*L/4;
            kin(i+d,i+1+d)=kin(i+1+d,i+d)=pow((1.0/lambda),(i-1)/2.0);
        }


    for(int i=0; i<L;i++)
        for(int j=0; j<L;j++)
            if( fabs( kin(i,j) ) > 1e-13 )
                h += Fermi(i,L,true)*Fermi(j,L,false)*kin(i,j);

    //--------------------int imp1 canal1------------------------------------

    h += Fermi(0,L,true)*Fermi(L/4,L,false)*Fermi(L/4+1,L,true)*Fermi(1,L,false)*(0.5)*Tk1;
    h += Fermi(L/4,L,true)*Fermi(0,L,false)*Fermi(1,L,true)*Fermi(L/4+1,L,false)*(0.5)*Tk1;

    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(1,L,true)*Fermi(1,L,false)*(0.25)*Tk1;
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(L/4+1,L,true)*Fermi(L/4+1,L,false)*(-0.25)*Tk1;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(1,L,true)*Fermi(1,L,false)*(-0.25)*Tk1;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(L/4+1,L,true)*Fermi(L/4+1,L,false)*(0.25)*Tk1;


    //--------------------int imp2 canal2------------------------------------

    h += Fermi(L/2,L,true)*Fermi(3*L/4,L,false)*Fermi(3*L/4+1,L,true)*Fermi(L/2+1,L,false)*(0.5)*Tk2;
    h += Fermi(3*L/4,L,true)*Fermi(L/2,L,false)*Fermi(L/2+1,L,true)*Fermi(3*L/4+1,L,false)*(0.5)*Tk2;

    h += Fermi(L/2,L,true)*Fermi(L/2,L,false)*Fermi(L/2+1,L,true)*Fermi(L/2+1,L,false)*(0.25)*Tk2;
    h += Fermi(L/2,L,true)*Fermi(L/2,L,false)*Fermi(3*L/4+1,L,true)*Fermi(3*L/4+1,L,false)*(-0.25)*Tk2;
    h += Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*Fermi(L/2+1,L,true)*Fermi(L/2+1,L,false)*(-0.25)*Tk2;
    h += Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*Fermi(3*L/4+1,L,true)*Fermi(3*L/4+1,L,false)*(0.25)*Tk2;

    //-----------------------Hund impurezas------------------------------------

    h += Fermi(0,L,true)*Fermi(L/4,L,false)*Fermi(3*L/4,L,true)*Fermi(L/2,L,false)*(-0.5)*J;
    h += Fermi(L/4,L,true)*Fermi(0,L,false)*Fermi(L/2,L,true)*Fermi(3*L/4,L,false)*(-0.5)*J;

    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*(-0.25)*J;
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*(0.25)*J;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*(0.25)*J;
    h += Fermi(L/4,L,true)*Fermi(L/4,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*(-0.25)*J;


    return h.toMPS().Sweep();
}

MPO Ham2Cdesac(int L, double BF)
{
    int m=2;
    double lambda=6;
    double Ec=5;
    double U=4;
    double J=0.0;
    double B=BF;
    double epsilon=-2;
    //---interaccion-QD-bandas
    double V=0.6;
    double V1=1.5*V;
    double V2=-V;

    MPSSum h(m,MatSVDFixedTol(1e-12));

    //--------------------------------------------------------------------
    //-------------------------impurezas----------------------------------
    //-----------------------CANALES-ESTRELLA-----------------------------

    mat kin = zeros<mat>(L,L);

    //------------LVL1UP--BAÑO-UP---LVL1DOWN--BAÑO-DOWN---------------
    //------------LVL2UP--BAÑO-UP---LVL2DOWN--BAÑO-DOWN---------------

    //-------------ENERGIAS-DIAGONALES-NIVELES-QD---------------------

    kin(0,0)=kin(L/2,L/2)=epsilon-0.5*B;
    kin(L/4,L/4)=kin(3*L/4,3*L/4)=epsilon+0.5*B;


    //---------------------------canales------------------------------
    //-----------------------hoppings NRG-----------------------------

    for(int i=1;i<L/4-1;i++)
        for(int j=0;j<4;j++)
        {
            int d=j*L/4;
            kin(i+d,i+1+d)=kin(i+1+d,i+d)=pow((1.0/lambda),(i-1)/2.0);
        }

    //---------------------------hibridizacion------------------------

    kin(0,1)=V1;
    kin(1,0)=V1;
    kin(L/4,L/4+1)=V1;
    kin(L/4+1,L/4)=V1;
    kin(L/2,L/2+1)=V2;
    kin(L/2+1,L/2)=V2;
    kin(3*L/4,3*L/4+1)=V2;
    kin(3*L/4+1,3*L/4)=V2;

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

    //-----------------Termino---U--------------------------------------
    h += Fermi(0,L,true)*Fermi(0,L,false)*Fermi(L/4,L,true)*Fermi(L/4,L,false)*U;
    h += Fermi(L/2,L,true)*Fermi(L/2,L,false)*Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false)*U;


    return h.toMPS().Sweep();
}

MPO HamSIAM(int L, double BF)
{
    int m=2;
    double lambda=2;
    double Ec=5;
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

MPO HamSIAMNoNRG(int L, double BF)
{
    int m=2;
    double Ec=5;
    double U=4;
    double B=BF;
    double epsilon=-2;
    //---interaccion-QD-bandas
    double V=0.6;
    double V1=1.5*V;


    MPSSum h(m,MatSVDFixedTol(1e-12));

    //-------------------------impurezas----------------------------------
    //-----------------------CANALES-ESTRELLA-----------------------------

    mat kin = zeros<mat>(L,L);

    //-------------ENERGIAS-DIAGONALES-NIVELES-QD---------------------

    kin(0,0)=epsilon-0.5*B;
    kin(L/2,L/2)=epsilon+0.5*B;

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

//-----------Reflexion-----------------------------------------------

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
    for(int i=0;i<2; i++)
        npart += Fermi(i*L/4,L,true)*Fermi(i*L/4,L,false);

    npart += Fermi(3*L/4-1,L,true)*Fermi(3*L/4-1,L,false);
    npart += Fermi(L-1,L,true)*Fermi(L-1,L,false);
    return npart.toMPS();
}
MPO NImp1up(int L)
{
    return Fermi(0,L,true)*Fermi(0,L,false);
}
MPO NImp1down(int L)
{
    return Fermi(L-1,L,true)*Fermi(L-1,L,false) ;
}
MPO NImp2up(int L)
{
    return Fermi(L/4,L,true)*Fermi(L/4,L,false) ;
}
MPO NImp2down(int L)
{
    return Fermi(3*L/4-1,L,true)*Fermi(3*L/4-1,L,false) ;
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

void CalculateNiNj(const Parameters par)
{
    MPS gs;
    gs.Load("gs.dat");
    ofstream out("ninj.dat");
    int L=par.length;
    vector<int> pos={0,L/4,3*L/4-1,L-1};
    for(int i:pos)
    {
        for(int j:pos)
        {
            MPO rr=Fermi(i,L,true)*Fermi(i,L,false)*Fermi(j,L,true)*Fermi(j,L,false);
            double ninj=Superblock({&gs,&rr,&gs}).value();
            out<<ninj<<" ";
        }
        out<<endl;
    }
}

//-----------Traslacion----------------------------------------------

// MPO NParticle(int L)
// {
//     int m=4;
//     MPSSum npart(m,MatSVDFixedTol(1e-13));
//     for(int i=0;i<L; i++)
//         npart += Fermi(i,L,true)*Fermi(i,L,false) ;
//     return npart.toMPS();
// }
// MPO NImp1up(int L)
// {
//     int m=1;
//     MPSSum npart(m,MatSVDFixedTol(1e-13));
//     npart += Fermi(0,L,true)*Fermi(0,L,false) ;
//     return npart.toMPS();
// }
// // MPO NImp1down(int L)
// // {
// //     int m=1;
// //     MPSSum npart(m,MatSVDFixedTol(1e-13));
// //     npart += Fermi(L/4,L,true)*Fermi(L/4,L,false) ;
// //     return npart.toMPS();
// // }
// MPO NImp2up(int L)
// {
//     int m=1;
//     MPSSum npart(m,MatSVDFixedTol(1e-13));
//     npart += Fermi(L/2,L,true)*Fermi(L/2,L,false) ;
//     return npart.toMPS();
// }
// // MPO NImp2down(int L)
// // {
// //     int m=1;
// //     MPSSum npart(m,MatSVDFixedTol(1e-13));
// //     npart += Fermi(3*L/4,L,true)*Fermi(3*L/4,L,false) ;
// //     return npart.toMPS();
// // }
// MPO NImp(int L)
// {
//     int m=2;
//     MPSSum npart(m,MatSVDFixedTol(1e-13));
//     for(int i=0;i<2; i++)
//         npart += Fermi(i*L/2,L,true)*Fermi(i*L/2,L,false) ;
//     return npart.toMPS();
// }
// //------------------original--------------------------------------
// // MPO NImp(int L)
// // {
// //     int m=4;
// //     MPSSum npart(m,MatSVDFixedTol(1e-13));
// //     for(int i=0;i<4; i++)
// //         npart += Fermi(i*L/4,L,true)*Fermi(i*L/4,L,false) ;
// //     return npart.toMPS();
// // }
//
// void CalculateNi(const Parameters par)
// {
//     MPS gs;
//     gs.Load("gs.dat");
//     ofstream out("ni.dat");
//     int L=par.length;
//     for(int i=0; i<L; i++)
//     {
//         MPO rr=Fermi(i,L,true)*Fermi(i,L,false);
//         double ni=Superblock({&gs,&rr,&gs}).value();
//         out<<i<<" "<<ni<<endl;
//     }
// }
//
// void CalculateNiNj(const Parameters par)
// {
//     MPS gs;
//     gs.Load("gs.dat");
//     ofstream out("ninj.dat");
//     int L=par.length;
// //    vector<int> pos={0,L/4,L/2,3*L/4};
//     vector<int> pos={0,L/2};
//     for(int i:pos)
//     {
//         for(int j:pos)
//         {
//             MPO rr=Fermi(i,L,true)*Fermi(i,L,false)*Fermi(j,L,true)*Fermi(j,L,false);
//             double ninj=Superblock({&gs,&rr,&gs}).value();
//             out<<ninj<<" ";
//         }
//         out<<endl;
//     }
// }

//---------------------------- Test DMRG basico -------------------------------------------

void TestDMRGBasico(const Parameters &par)
{
    int len=par.length;
    auto op=HamICTP3(len,par.BF); op.Sweep(); op.PrintSizes("HamICTP3=");
    op.decomposer=MatQRDecomp;
    auto nop=NParticle(len),nimp=NImp(len);
    auto nimp1up=NImp1up(len),nimp2up=NImp2up(len);
    auto nimp1down=NImp1down(len),nimp2down=NImp2down(len);

    DMRG_krylov_gs sol(op,par.m,par.nkrylov);
    //sol.nsite_gs=1;
    sol.DoIt_gs();
    for(int k=0;k<par.nsweep;k++)
    {
        sol.DoIt_res(par.nsweep_resid);
        std::cout<<"sweep "<<k+1<< "  error="<<sol.error<<" --------------------------------------\n";
        sol.reset_states();
        sol.DoIt_gs();
        Superblock np({&sol.gs[0],&nop,&sol.gs[0]});
        Superblock ni({&sol.gs[0],&nimp,&sol.gs[0]});
        Superblock ni1u({&sol.gs[0],&nimp1up,&sol.gs[0]});
        Superblock ni1d({&sol.gs[0],&nimp1down,&sol.gs[0]});
        Superblock ni2u({&sol.gs[0],&nimp2up,&sol.gs[0]});
        Superblock ni2d({&sol.gs[0],&nimp2down,&sol.gs[0]});
        cout<<"nImp1up="<<ni1u.value()<<" nImp2up="<<ni2u.value()<< endl;
        cout<<"nImp1dw="<<ni1d.value()<<" nImp2dw="<<ni2d.value()<< endl;
        cout<<"nImp="<<ni.value()<<" nT="<<np.value()<<endl;
    }
    ofstream out("gs.dat");
    sol.gs[0].Save(out);
}

//---------------------------- Test DMRG-CV ---------------------------------------------

vector<cmpx> ReadWFile(const string& name)
{
    vector<cmpx> ws;
    double x,y;
    complex<double> z;
    ifstream in(name);
    if (!in.is_open())
        throw invalid_argument("TestDMRGCV_Impuritynn::ReadWFile file not found");
    while (!in.eof())
    {
        in>>x>>y; z={x,y};
        ws.push_back(z);
        if (in.eof()) {ws.pop_back(); break;}
        in.ignore(1000,'\n');
    }
    return ws;
}

vector<cmpx> LocalInterval(const vector<cmpx>& ws, int dw,int iid)
{
    vector<cmpx> res;
    int i0=0; //ws.size()/2;
    if ( iid*dw>=int(ws.size()/2) ) i0=1;
    for(int i=i0+dw*iid; i<i0+dw*(iid+1); i++)
        res.push_back(ws[i]);
    return res;
}

double EtaMax(const vector<cmpx>& ws)
{
    double etam=ws[0].imag();
    for(const cmpx& z:ws)
        if (z.imag()>etam) etam=z.imag();
    return etam;
}

void TestDMRGCV(const Parameters& param,int id, int n_id)
{
    int len=param.length, m=param.m;

    auto op=HamSIAMNoNRG(len,param.BF); op.Sweep(); op.PrintSizes("HamSIAMNoNRG=");
    op.decomposer=MatQRDecomp;

    MPS gs;
    ifstream in("gs.dat");
    gs.Load(in);
    double ener=Superblock({&gs,&op,&gs}).value();

    vector<cmpx> wsG=ReadWFile("ws.dat");
    int dw=wsG.size()/n_id;
    auto ws=LocalInterval(wsG,dw,id-1);
    if ( id==n_id/2+1)
    {
        cmpx z=wsG[wsG.size()/2];
        ws.insert(ws.begin(),z);
    }
    cout<<"\nener="<<ener;
    cout<<" length="<<param.length<<" nsweeps="<<param.nsweep<<" m="<<param.m<<endl;
    cout<<"interval id="<<id<<" w1="<<ws.front()<<" w2="<<ws.back()<<endl;
    double w1=ws.front().real();
    double w2=ws.back().real();
    double eta1=std::min( ws.front().imag()*param.etaFactor , EtaMax(wsG) );
    double eta2=std::min( ws.back() .imag()*param.etaFactor , EtaMax(wsG) );

    MPS a= (param.opType=='C')?Fermi(param.op1Pos,len,true)*gs
                             :Fermi(param.op1Pos,len,false)*gs;
    a.Canonicalize();
    DMRG_0_cv solcv(op,m,a,ener,{w1,w2},{eta1,eta2});
    std::cout<<"\n\n\nstarting CV\n\n";
    for(int k=0;k<param.nsweep;k++)
    {
        std::cout<<"sweep "<<k+1<<"\n";
        for(auto p:MPS::SweepPosSec(len))
        {
            solcv.SetPos(p);
            solcv.Solve();
            solcv.Print();
        }
    }
    MPS b= (param.opType=='C')?Fermi(param.op2Pos,len,true)*gs
                             :Fermi(param.op2Pos,len,false)*gs;
    b.Canonicalize();
    auto green=solcv.Green(b,ws);
    ofstream out( string("green")+
                  param.opType+
                  to_string(param.op1Pos)+
                  to_string(param.op2Pos)+
                  to_string(id)+
                  ".dat");
    for(uint i=0;i<ws.size();i++)
        out<<ws[i].real()<<" "<<ws[i].imag()<<" "<<green[i].real()<<" "<<green[i].imag()<<"\n";
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
        CalculateNiNj(param);
    }
    else if (argc==4) // ./a.out parameters.dat <interval id> <n intervals>
    {
        Parameters param;
        param.ReadParameters(argv[1]);
        TestDMRGCV(param,atoi(argv[2]),atoi(argv[3]));
    }
    cout<<"\nDone in "<<difftime(time(NULL),t0)<<"s"<<endl;
    return 0;
}
