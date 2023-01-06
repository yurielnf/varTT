#include <iostream>
#include "dmrg1_wse_gs.h"
#include "dmrg1_wse_gs.h"
#include "parameters.h"
#include "hamhall.h"
#include "freefermions.h"

#include <fftw3.h>
// en terminal: gcc main.c -lfftw3 -lm
// Ver http://www.fftw.org/fftw3_doc/

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

MPO HamiltonianHall(const Parameters &par)
{
    int len=par.length;
    auto hh=HamHall(len);
    hh.in_W=par.hallW_file;
    hh.mu=par.mu;
    hh.periodic=par.periodic;
    hh.Load();
    hh.d_cut_exp=len; hh.d_cut_Fourier=len; hh.tol=1e-6;
    auto mpo=hh.Hamiltonian(); mpo.Sweep(); mpo.PrintSizes("HamHall=");
    mpo.decomposer=MatQRDecomp;
    mpo.Save("ham.dat");
    return mpo;
}


void TestDMRGBasico(const Parameters &par)
{
    int len=par.length;
    MPO ham=HamiltonianHall(par);
    ham.PrintSizes();
    auto nop=NParticle(len);
    auto eh_op=ElectronHoleMPO(len);
    MPO Proj=eh_op+MPOIdentity(len,2), gss, gsr, gsrr;
    Proj.Canonicalize();
    DMRG1_wse_gs sol(ham,par.m);
    sol.tol_diag=1e-8;
    for(int k=0;k<par.nsweep;k++)
    {
        bool use_arpack= k==par.nsweep-1;
        if (use_arpack) sol.tol_diag=1e-10;
        use_arpack=true;
        std::cout<<"sweep "<<k+1<<" --------------------------------------\n";
        for(auto p : MPS::SweepPosSec(len))
        {
            sol.SetPos(p);
            sol.Solve(use_arpack);
            if ((p.i+1) % (len/10) ==0) sol.Print();
        }
        gss=(par.mu!=0)? sol.gs : MPO_MPS{Proj,sol.gs}.toMPS(2*par.m).Normalize();
        {// symmetrize under reflexion
            MPSSum sr(2*par.m,MatSVDFixedDim(2*par.m));
            sr += sol.gs;
            MPS xr=sol.gs.Reflect(); xr.SetPos(sol.gs.pos);
            sr += xr;
            gsr=sr.toMPS().Canonicalize().Normalize();
            gsrr=gsr.Reflect(); gsrr.SetPos(sol.gs.pos);
        }
        Superblock np({&gss,&nop,&gss});
        Superblock eh({&gss,&eh_op,&gss});
        Superblock rf({&gsrr,&gsr});
        cout<<" nT="<<np.value()<<", eh="<<eh.value()<<" Reflect="<<rf.value()<<endl;
    }
    gsr.Save("gs.dat");
}

void SimetrizeOldGs()
{
    MPS gs;
    gs.Load("gs.dat");
    MPS Proj=ElectronHoleMPO(gs.length)+MPOIdentity(gs.length,2);
    Proj.Canonicalize();
    MPS gss=MPO_MPS{Proj,gs}.toMPS(gs.m).Normalize();
    gss.Save("gs.dat");
}
void SimetrizeOldGsR()
{
    MPS gs;
    gs.Load("gs.dat");
    gs.PrintSizes("gs=");
    cout<<"pos="<<gs.pos.i<<" "<<gs.pos.vx<<"\n";
    MPSSum sr(gs.m,MatSVDFixedTol(1e-10));
    sr += gs;
    MPS xr=gs.Reflect();
    cout<<"pos="<<xr.pos.i<<" "<<xr.pos.vx<<"\n";
    xr.SetPos(gs.pos);
    cout<<"pos="<<xr.pos.i<<" "<<xr.pos.vx<<"\n";
    sr += xr;
    auto gsr=sr.toMPS();//.Sweep();//.Canonicalize().Normalize();
    gsr.Save("gs.dat");
}

void CalculaFFT(double ft[], size_t N, double Pk[])
{
    fftw_complex *in, *out;
    fftw_plan p;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for(size_t i=0;i<N;i++)
    {
        in[i][0]=ft[i];
        in[i][1]=0;
    }

    fftw_execute(p); /* repeat as needed */

    for(size_t i=0;i<N;i++)
        Pk[i]=out[i][0]*out[i][0]+out[i][1]*out[i][1];

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
}

void ExportSTable(string filename,const stdvec& qs,const TensorD& s)
{
    ofstream out(filename.c_str());
    out<<"r ";
    for(auto q:qs) out<<"q="<<q<<" ";
    out<<endl;
    for(int i=0;i<s.dim[0];i++)
    {
        out<<i<<" ";
        for(int j=0;j<qs.size();j++)
            out<<s[{i,j}]<<" ";
        out<<endl;
    }
}

void CalculateS(const Parameters &par)
{
    MPS gs;
    gs.Load("gs.dat");
    const auto &qs=par.renyi_q;
    TensorD s({ gs.length/2, int(qs.size()) });
    for(int i=0;i<gs.length/2;i++)
    {
        gs.SetPos({i,1});
        TensorD rho=gs.C*gs.C.t();
        TensorD eval=rho.EigenDecomposition(1).at(1);
        for(int j=0;j<qs.size();j++)
            s[{i,j}]=EntropyRenyi(eval.data(),eval.size(),qs[j]);
    }
    TensorD rsp({s.dim[0]-1, s.dim[1]}), sk=rsp;
    for(int i=0;i<gs.length/2-1;i++)
        for(int j=0;j<qs.size();j++)
            rsp[{i,j}]=(i+1)*( s[{i+1,j}]-s[{i,j}] );   //r*S'(r)
    for(int j=0;j<qs.size();j++)
        CalculaFFT(&rsp[{0,j}],rsp.dim[0],&sk[{0,j}]);

    ExportSTable("entropy.dat",qs,s);
    ExportSTable("entropy_rSp.dat",qs,rsp);
    ExportSTable("entropy_fourier.dat",qs,sk);
}

void CalculaFFT(cmpx in[], cmpx out[], int N, bool direct)
{
    int sign=direct?FFTW_FORWARD:FFTW_BACKWARD;
    fftw_complex *inp =reinterpret_cast<fftw_complex*>(in);
    fftw_complex *outp=reinterpret_cast<fftw_complex*>(out);
    fftw_plan p= fftw_plan_dft_1d(N, inp, outp, sign, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);
}

void CalculateRhoRhok(const Parameters par)
{
    MPS gs;
    gs.Load("gs.dat");
    ofstream out("rho_rho.dat");
    int N=par.length, x1=N/2;
    //    double Ly=10, factor=2*M_PI/Ly;
    for(int mn=-N/2; mn<N/2; mn++)
    {
        int m=mn<0 ? mn+N : mn;
        vector<cmpx> corr(N);
        for(int k=0;k<N;k++)
        {
            MPO rr=Fermi(x1,N,true)*Fermi((x1+m)%N,N,false)*
                    Fermi((x1+k+m)%N,N,true)*Fermi((x1+k)%N,N,false);
            corr[k]=Superblock({&gs,&rr,&gs}).value();
        }
        vector<cmpx> ck(N);
        CalculaFFT(corr.data(),ck.data(),N,true);

        for(int in=-N/2;in<N/2;in++)
        {
            int i=in<0 ? in+N : in;
            out<<mn<<" "<<in<<" "<<2*M_PI*in/N<<" "<<corr[i].real()<<" "<<ck[i].real()<<" "<<ck[i].imag()<<endl;
        }
        out<<endl;
    }
}

void CalculateNi(const Parameters par)
{
    MPS gs;
    gs.Load("gs.dat");
    ofstream out("ni.dat");
    int N=par.length;
    for(int i=0; i<N; i++)
    {
        MPO rr=Fermi(i,N,true)*Fermi(i,N,false);
        double ni=Superblock({&gs,&rr,&gs}).value();
        out<<i+1<<" "<<ni<<endl;
    }
}

void CalculateNiN0(const Parameters par)
{
    MPS gs;
    gs.Load("gs.dat");
    ofstream out("nin0.dat");
    int N=par.length;
    MPO rr0=Fermi(N/2-1,N,true)*Fermi(N/2-1,N,false);
    double n_0=Superblock({&gs,&rr0,&gs}).value();
    for(int i=0; i<N; i++)
    {
        MPO rri=Fermi(i,N,true)*Fermi(i,N,false);
        double n_i=Superblock({&gs,&rri,&gs}).value();
        MPO rr=Fermi(i,N,true)*Fermi(i,N,false)*Fermi(N/2-1,N,true)*Fermi(N/2-1,N,false);
        double nin0=Superblock({&gs,&rr,&gs}).value()-n_i*n_0;
        out<<i+1<<" "<<nin0<<endl;
    }
}


int main(int argc, char *argv[])
{
    cout << "Hello World!" << endl;
    std::cout<<std::setprecision(15);
    time_t t0=time(nullptr);
    srand(t0);

    if (argc==3 && string(argv[2])=="hall")  // ./a.out parameters.dat hall
    {
        Parameters param;
        param.ReadParameters(argv[1]);
        TestDMRGBasico(param);
        //        CalculateNi(param);
        //        CalculateNiN0(param);
    }
    else if (argc==4 && string(argv[2])=="hall"     // ./a.out parameters.dat hall ham
             && string(argv[3])=="ham")
    {
        Parameters param;
        param.ReadParameters(argv[1]);
        HamiltonianHall(param);
    }
    else if (argc==4 && string(argv[2])=="hall"
             && string(argv[3])=="renyi"         )  // ./a.out parameters.dat hall renyi
    {
        Parameters param;
        param.ReadParameters(argv[1]);
        CalculateS(param);
    }
    else if (argc==4 && string(argv[2])=="hall"
             && string(argv[3])=="nin0"         )  // ./a.out parameters.dat hall nin0
    {
        Parameters param;
        param.ReadParameters(argv[1]);
        CalculateNi(param);
        CalculateNiN0(param);
    }
    else if (argc==4 && string(argv[2])=="hall"
             && string(argv[3])=="rhorho"         )  // ./a.out parameters.dat hall rhorho
    {
        Parameters param;
        param.ReadParameters(argv[1]);
        CalculateRhoRhok(param);
    }
    else if (argc==4 && string(argv[2])=="hall"
             && string(argv[3])=="eh"         )  // ./a.out parameters.dat hall eh
    {
        SimetrizeOldGs();
    }
    else if (argc==4 && string(argv[2])=="hall"
             && string(argv[3])=="Reflect"         )  // ./a.out parameters.dat hall Reflect
    {
        SimetrizeOldGsR();
    }
    else if (argc==4 && string(argv[2])=="hall"
             && string(argv[3])=="free1d"         )  // ./a.out parameters.dat hall free1d
    {
        Parameters param;
        param.ReadParameters(argv[1]);
        TestRenyiLibre1D(param);
    }
    else if (argc==4 && string(argv[2])=="hall"
             && string(argv[3])=="free2d"         )  // ./a.out parameters.dat hall free1d
    {
        Parameters param;
        param.ReadParameters(argv[1]);
        TestRenyiLibre2D(param);
    }
    else cout<<"./a.out <arg>";

    cout<<"\nDone in "<<difftime(time(nullptr),t0)<<"s"<<endl;
    return 0;
}
