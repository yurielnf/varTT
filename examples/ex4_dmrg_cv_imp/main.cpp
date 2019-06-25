#include <iostream>
#include"dmrg_res_gs.h"
#include"dmrg1_wse_gs.h"
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

MPO Ham2CK1(int L)
{
    int m=1000;
    double lambda=6;
    double V=1.13;
    double U1=30;
    double U2=10;
    double Jh=10;
    double e=-2;
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

//---------------------------- Test DMRG basico -------------------------------------------

void TestDMRGBasico(const Parameters &param)
{
    int len=param.length, m=param.m, mMax=m;

    auto op=Ham2CK1(len); op.Sweep(); op.PrintSizes("Hamtb=");
    op.decomposer=MatChopDecompFixedTol(0);
    auto nop=NParticle(len),nimp=NImp(len);
    DMRG_0_gs sol(op,m,mMax,1.0);
    for(int k=0;k<param.nsweeps;k++)
    {
        std::cout<<"sweep "<<k+1<< "  error="<<sol.error<<" --------------------------------------\n";
        sol.DoIt_gs();
        Superblock np({&sol.gs,&nop,&sol.gs});
        Superblock ni({&sol.gs,&nimp,&sol.gs});
        cout<<"nImp="<<ni.value()<<" nT="<<np.value()<<endl;
        if (k<param.nsweeps-1) sol.DoIt_res();
    }
    ofstream out("gs.dat");
    sol.gs.Save(out);
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

    auto op=Ham2CK1(len); op.Sweep(); op.PrintSizes("Ham=");
    op.decomposer=MatChopDecompFixedTol(0);

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
    cout<<" length="<<param.length<<" nsweeps="<<param.nsweeps<<" m="<<param.m<<endl;
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
    for(int k=0;k<param.nsweeps;k++)
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
