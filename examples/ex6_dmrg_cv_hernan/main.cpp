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


cx_mat GreenOfHamiltonian0(const mat& ham,std::complex<double> z,const vector<int>& pos)
{
    int n=pos.size();
    mat evec;
    vec eval;
    eig_sym(eval,evec,ham);
    cx_mat g=cx_mat(n,n,fill::zeros);
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            for(int k=0;k<ham.n_rows;k++)
                g(i,j)+=evec(pos[i],k)*evec(pos[j],k) / ( z-eval(k) );
    return g;
}

MPO HamTb1(int L,bool periodic)
{
    const int m=1;
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
    int m=1;
    MPSSum npart(m,MatSVDFixedTol(1e-13));
    for(int i=0;i<L; i++)
        npart += Fermi(i,L,true)*Fermi(i,L,false) ;
    return npart.toMPS();
}


//---------------------------- Test DMRG basico -------------------------------------------


void TestDMRGBasico(const Parameters &par)
{
    int len=par.length;
    auto op=HamTb1(len,par.periodic); op.Sweep(); op.PrintSizes("Hamtb=");
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
    cout<<"exact="<<ExactEnergyTB(len,len/2,par.periodic);
    ofstream out("gs.dat");
    sol.gs[0].Save(out);
}

/*
void TestDMRGBasico(const Parameters &par)
{
    int len=par.length;
    auto op=HamTb1(len,par.periodic); op.Sweep(); op.PrintSizes("Hamtb=");
    op.decomposer=MatQRDecomp;
    auto nop=NParticle(len);
    DMRG1_wse_gs sol(op,par.m);
    sol.tol_diag=1e-4;
    for(int k=0;k<par.nsweep;k++)
    {
        if (k==par.nsweep-1) sol.tol_diag=1e-10;
        for(auto p : MPS::SweepPosSec(op.length))
        {
            sol.SetPos(p);
            sol.Solve();
            if ((p.i+1) % (op.length/10) ==0) sol.Print();
        }
        std::cout<<"sweep "<<k+1<<" --------------------------------------\n";
        cout<<" nT="<<Superblock({&sol.gs,&nop,&sol.gs}).value()<<endl;
        cout<<"exact="<<ExactEnergyTB(len,len/2,par.periodic)<<"\n";
    }
    ofstream out("gs.dat");
    sol.gs.Save(out);
}
*/

//---------------------------- Test DMRG-CV ---------------------------------------------

vector<cmpx> ReadWFile(const string& name)
{
    vector<cmpx> ws;
    double x,y;
    complex<double> z;
    ifstream in(name);
    if (!in.is_open())
        throw invalid_argument("TestDMRGCV::ReadWFile file not found");
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

void TestDMRGCV(const Parameters& param,int id, int n_id)
{
    int len=param.length, m=param.m;

    auto op=HamTb1(len,false); op.Sweep(); op.PrintSizes("Ham=");
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

    if (false)
    {
        ofstream out("greenE.dat");
        int L=100;
        mat h(L,L,fill::zeros);
        h.diag(-1).fill(1);
        h.diag(1).fill(1);
        cmpx z1(-2.5,0.01), z2(2.5,0.01);
        int n=501;
        cmpx dz=(z2-z1)/(n-1.0);
        for(int i=0;i<n;i++)
        {
            auto z=z1+dz*(1.0*i);
            cmpx g=GreenOfHamiltonian0(h,z,{L/2})(0,0);
            out<<z.real()<<" "<<z.imag()<<" "<<g.real()<<" "<<g.imag()<<endl;
        }
        return 0;
    }

    if (argc==3 && string(argv[2])=="gs")  // ./a.out parameters.dat gs
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
