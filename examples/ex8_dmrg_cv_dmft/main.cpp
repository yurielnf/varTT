#include <iostream>
#include"dmrg_res_gs.h"
#include"dmrg1_wse_gs.h"
#include"dmrg_krylov_gs.h"
#include"dmrg_res_cv.h"
#include"parameters.h"
#include"hamnn.h"

#include<armadillo>

using namespace std;
using namespace arma;


MPO NParticle(int L)
{
    int m=4;
    MPSSum npart(m,MatSVDFixedTol(1e-13));
    for(int i=0;i<L; i++)
        npart += Fermi(i,L,true)*Fermi(i,L,false) ;
    return npart.toMPS();
}
void CalculateNi()
{
    MPS gs;
    gs.Load("gs.dat");
    ofstream out("ni.dat");
    int L=gs.length;
    for(int i=0; i<L; i++)
    {
        MPO rr=Fermi(i,L,true)*Fermi(i,L,false);
        double ni=Superblock({&gs,&rr,&gs}).value();
        out<<i<<" "<<ni<<endl;
    }
}

//---------------------------- Test DMRG basico -------------------------------------------

void TestDMRGBasico(const Parameters &par)
{
    int len=par.length*2;
    auto sysNN=HamNN(len); sysNN.Load();
    auto op=sysNN.toMPO();//
    op.Sweep(); op.PrintSizes("Ham=");
    op.decomposer=MatQRDecomp;
    auto nop=NParticle(len);
    //DMRG_krylov_gs sol(op,par.m,par.nkrylov);
//    sol.DoIt_gs();
    DMRG1_wse_gs sol(op,par.m);
    sol.tol_diag=1e-4;
    for(int k=0;k<par.nsweep;k++)
    {
//        sol.DoIt_res(par.nsweep_resid);
//        std::cout<<"sweep "<<k+1<< "  error="<<sol.error<<" --------------------------------------\n";
//        sol.reset_states();
//        sol.DoIt_gs();
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
    //cout<<"exact="<<ExactEnergyTB(len,len/2,false);
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

vector<cmpx> LocalInterval(const vector<cmpx>& ws, int dw, int iid)
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

void SetExcitation(const Parameters &par,vector<int> impPos,int L,
                   const MPS &gs,MPS &a,MPS &b)
{
    if (par.opType=='C')
    {
        if (par.opProj=="hd") //holon-doblon
        {
            int nI=impPos.size()/2;
            a=Fermi(impPos[0],L,true)*
                    Fermi(impPos[nI],L,true)*Fermi(impPos[nI],L,false)*
                    Fermi(impPos[1],L,false)*Fermi(impPos[1],L,true)*
                    Fermi(impPos[1+nI],L,false)*Fermi(impPos[1+nI],L,true)*gs;
            b=Fermi(impPos[0],L,true)*gs;
        }
        else if (par.opProj=="hdhd") //holon-doblon
        {
            int nI=impPos.size()/2;
            a=Fermi(impPos[0],L,true)*
                    Fermi(impPos[nI],L,true)*Fermi(impPos[nI],L,false)*
                    Fermi(impPos[1],L,false)*Fermi(impPos[1],L,true)*
                    Fermi(impPos[1+nI],L,false)*Fermi(impPos[1+nI],L,true)*gs;
            b=a;
        }
        else if (par.opProj=="bwneg") //banda w<0 y otras
        {
            int nI=impPos.size()/2;
            a=Fermi(impPos[0],L,true)*
                    Fermi(impPos[nI],L,false)*Fermi(impPos[nI],L,true)*
                    Fermi(impPos[1],L,false)*Fermi(impPos[1],L,true)*
                    Fermi(impPos[1+nI],L,false)*Fermi(impPos[1+nI],L,true)*gs;
            b=Fermi(impPos[0],L,true)*gs;
        }
        else if (par.opProj=="bb") //banda b
        {
            int nI=impPos.size()/2;
            a=Fermi(impPos[0],L,true)*
                    Fermi(impPos[nI],L,false)*Fermi(impPos[nI],L,true)*
                    Fermi(impPos[1],L,true)*Fermi(impPos[1],L,false)*
                    Fermi(impPos[1+nI],L,false)*Fermi(impPos[1+nI],L,true)*gs;
            b=Fermi(impPos[0],L,true)*gs;
        }
        else
        {
            a=Fermi(impPos[par.op1Pos],L,true)*gs;
            b=Fermi(impPos[par.op2Pos],L,true)*gs;
        }
    }
    else //'D'
    {
        if (par.opProj=="hd")
        {
            int nI=impPos.size()/2;
            a=Fermi(impPos[0],L,false)*
                    Fermi(impPos[nI],L,true)*Fermi(impPos[nI],L,false)*
                    Fermi(impPos[1],L,false)*Fermi(impPos[1],L,true)*
                    Fermi(impPos[1+nI],L,false)*Fermi(impPos[1+nI],L,true)*gs;
            b=Fermi(impPos[0],L,false)*gs;
        }
        else if (par.opProj=="hdhd")
        {
            int nI=impPos.size()/2;
            a=Fermi(impPos[0],L,false)*
                    Fermi(impPos[nI],L,true)*Fermi(impPos[nI],L,false)*
                    Fermi(impPos[1],L,false)*Fermi(impPos[1],L,true)*
                    Fermi(impPos[1+nI],L,false)*Fermi(impPos[1+nI],L,true)*gs;
            b=a;
        }
        else if (par.opProj=="bwneg") //banda w<0
        {
            int nI=impPos.size()/2;
            a=Fermi(impPos[0],L,false)*
                    Fermi(impPos[nI],L,false)*Fermi(impPos[nI],L,true)*
                    Fermi(impPos[1],L,false)*Fermi(impPos[1],L,true)*
                    Fermi(impPos[1+nI],L,false)*Fermi(impPos[1+nI],L,true)*gs;
            b=Fermi(impPos[0],L,false)*gs;
        }
        else if (par.opProj=="bb") //banda b
        {
            int nI=impPos.size()/2;
            a=Fermi(impPos[0],L,false)*
                    Fermi(impPos[nI],L,false)*Fermi(impPos[nI],L,true)*
                    Fermi(impPos[1],L,true)*Fermi(impPos[1],L,false)*
                    Fermi(impPos[1+nI],L,false)*Fermi(impPos[1+nI],L,true)*gs;
            b=Fermi(impPos[0],L,false)*gs;
        }
        else
        {
            a=Fermi(impPos[par.op1Pos],L,false)*gs;
            b=Fermi(impPos[par.op2Pos],L,false)*gs;
        }
    }
    a.Canonicalize();
    b.Canonicalize();
}

void TestDMRGCV(const Parameters& par,int id, int n_id)
{
    int len=par.length*2, m=par.m;

//    auto op=HamTb1(len,false)
    auto sysNN=HamNN(len); sysNN.Load();
    auto hnn=sysNN.toMPO(); hnn.Sweep(); hnn.PrintSizes("Ham=");
    hnn.decomposer=MatQRDecomp;

    cout<<"calculando la green en pos: "<<sysNN.impPos[par.op1Pos]<<"; "<<sysNN.impPos[par.op1Pos]<<endl;

    MPS gs;
    ifstream in("gs.dat");
    gs.Load(in);
    double ener=Superblock({&gs,&hnn,&gs}).value();

    vector<cmpx> wsG=ReadWFile("hyb.dat");
    int dw=wsG.size()/n_id;
    auto ws=LocalInterval(wsG,dw,id-1);
    if ( id==n_id/2+1)
    {
        cmpx z=wsG[wsG.size()/2];
        ws.insert(ws.begin(),z);
    }
    cout<<"\nener="<<ener;
    cout<<" length="<<len<<" nsweeps="<<par.nsweep<<" m="<<par.m<<endl;
    cout<<"interval id="<<id<<" w1="<<ws.front()<<" w2="<<ws.back()<<endl;
    double w1=ws.front().real();
    double w2=ws.back().real();
    double eta1=std::min( ws.front().imag()*par.etaFactor , EtaMax(wsG) );
    double eta2=std::min( ws.back() .imag()*par.etaFactor , EtaMax(wsG) );


    MPS a,b;
    SetExcitation(par,sysNN.impPos,len,gs,a,b);

    DMRG_0_cv solcv(hnn,m,a,ener,{w1,w2},{eta1,eta2});
    solcv.useLanczosCv=par.useLanczosCv;
    std::cout<<"\n\n\nstarting CV\n\n";
    for(int k=0;k<par.nsweep;k++)
    {
        std::cout<<"sweep "<<k+1<<"\n";
        for(auto p:MPS::SweepPosSec(len))
        {
            solcv.SetPos(p);
            solcv.Solve();
            solcv.Print();
        }
    }

    auto green=solcv.Green(b,ws);
    ofstream out( string("green")+
                  par.opType+
                  to_string(par.op1Pos)+
                  to_string(par.op2Pos)+
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
        CalculateNi();
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
