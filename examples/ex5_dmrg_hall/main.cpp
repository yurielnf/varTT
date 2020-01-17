#include <iostream>
#include"dmrg_krylov_gs.h"
#include"dmrg_jacobi_davidson_gs.h"
#include"parameters.h"
#include"hamhall.h"
#include"freefermions.h"

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

void TestDMRGBasico(const Parameters &par)
{
    int len=par.length;
    auto hh=HamHall(len);
    hh.in_W=par.hallW_file;
    hh.mu=par.mu;
    hh.periodic=par.periodic;
    hh.Load();
    hh.d_cut_exp=len; hh.d_cut_Fourier=len; hh.tol=1e-6;
    auto op=hh.Hamiltonian(); op.Sweep(); op.PrintSizes("HamHall=");
    op.decomposer=MatQRDecomp;
    auto nop=NParticle(len), eh_op=ElectronHoleMPO(len);
    DMRG_krylov_gs sol(op,par.m,par.nkrylov);
    sol.nsite_gs=par.nsite_gs;
    sol.nsite_resid=par.nsite_resid;
//    sol.nsite_jd=par.nsite_jd;
    sol.error=1e-5;
    sol.DoIt_gs();
    MPS Proj=eh_op+MPOIdentity(len,2), gss; Proj.Canonicalize();
    for(int k=0;k<par.nsweep;k++)
    {
        sol.DoIt_res(par.nsweep_resid);
        std::cout<<"sweep "<<k+1<< "  error="<<sol.error<<" --------------------------------------\n";
        sol.reset_states();
        sol.DoIt_gs();
        gss=(par.mu!=0)? sol.gs[0] : MPO_MPS{Proj,sol.gs[0]}.toMPS(sol.m).Normalize();
        Superblock np({&gss,&nop,&gss});
        Superblock eh({&gss,&eh_op,&gss});
        cout<<" nT="<<np.value()<<", eh="<<eh.value()<<endl;
    }
    gss.Save("gs.dat");
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
        for(uint j=0;j<qs.size();j++)
            out<<s[{i,j}]<<" ";
        out<<endl;
    }
}

void CalculateS(const Parameters &par)
{
    MPS gs;
    gs.Load("gs.dat");
    const auto &qs=par.renyi_q;
    TensorD s({ gs.length/2, int(qs.size()) }), sk=s;
    for(int i=0;i<gs.length/2;i++)
    {
        gs.SetPos({i,1});
        TensorD rho=gs.C*gs.C.t();
        TensorD eval=rho.EigenDecomposition(1).at(1);
        for(uint j=0;j<qs.size();j++)
            s[{i,j}]=EntropyRenyi(eval.data(),eval.size(),qs[j]);
    }

    for(uint j=0;j<qs.size();j++)
    {
        TensorD sj=s.Subtensor(j);
        CalculaFFT(&s[{0,j}],s.dim[0],&sk[{0,j}]);
    }

    ExportSTable("entropy.dat",qs,s);
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
        CalculateNi(param);
    }
    else if (argc==4 && string(argv[2])=="hall"
             && string(argv[3])=="renyi"         )  // ./a.out parameters.dat hall renyi
    {
        Parameters param;
        param.ReadParameters(argv[1]);
        CalculateS(param);
    }
    else if (argc==4 && string(argv[2])=="hall"
             && string(argv[3])=="rhorho"         )  // ./a.out parameters.dat hall renyi
    {
        Parameters param;
        param.ReadParameters(argv[1]);
        CalculateNi(param);
        CalculateRhoRhok(param);
    }
    else if (argc==4 && string(argv[2])=="hall"
             && string(argv[3])=="eh"         )  // ./a.out parameters.dat hall eh
    {
        SimetrizeOldGs();
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
