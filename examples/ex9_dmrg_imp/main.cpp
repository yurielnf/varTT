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

/// L qubits, interaction U, hybridization V1
mat KinMatrixTB(int L,double U,double V1)
{
    int nbath=L/2-1;
    mat kinBath(nbath,nbath,fill::zeros); // matriz de energia cinetica del banho
    for(int i=0;i<nbath-1;i++)
        kinBath(i,i+1)=kinBath(i+1,i)=1;

    mat kin(L,L,fill::zeros); // energia cinetica total
    kin(nbath-1,nbath)=kin(nbath,nbath-1)=kin(nbath+1,nbath+2)=kin(nbath+2,nbath+1)=V1; // impurity at the center ---------->  oooxxooo for L=8
    for(int i=0;i<nbath;i++)
        for(int j=0;j<nbath;j++)
            kin(nbath-1-i,nbath-1-j)=kin(nbath+2+i,nbath+2+j)=kinBath(i,j);
    kin(L/2-1,L/2-1)=kin(L/2,L/2)=-0.5*U;
    return kin;
}


/// L qubits, interaction U, hybridization V1
MPO HamSiamTB(int L, double U, double V1)
{
    mat kin=KinMatrixTB(L,U,V1);

    MPSSum h(1,MatSVDFixedTol(1e-12));  // writing the Hamiltonian
    for(int i=0; i<L;i++)
        for(int j=0; j<L;j++)
            if( fabs( kin(i,j) ) > 1e-13 )
                h += Fermi(i,L,true)*Fermi(j,L,false)*kin(i,j);

    h += Fermi(L/2-1,L,true)*Fermi(L/2-1,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*U;
    return h.toMPS().Sweep();
}


/// L qubits, interaction U, hybridization V1
MPO HamSiamTBStar(int L, double U, double V1)
{
    mat k=KinMatrixTB(L,U,V1);

    // diagonalize the bath to compute the rotation
    vec eigval;
    mat eigvec;
    k.submat(0,0,L/2-2,L/2-2).print("bath=");
    eig_sym(eigval,eigvec,k.submat(0,0,L/2-2,L/2-2));

    mat rot(k.n_rows,k.n_cols,fill::eye);
    for(int i=0; i<L/2-1;i++)
        for(int j=0; j<L/2-1; j++)
            rot(L/2-2-i,L/2-2-j)=rot(i+L/2+1,j+L/2+1)=eigvec(i,j);              // configuration oooxxooo for L=8

    mat kin = rot.t()*k*rot;

    MPSSum h(1,MatSVDFixedTol(1e-12));  // writing the Hamiltonian
    for(int i=0; i<L;i++)
        for(int j=0; j<L;j++)
            if( fabs( kin(i,j) ) > 1e-13 )
                h += Fermi(i,L,true)*Fermi(j,L,false)*kin(i,j);

    h += Fermi(L/2-1,L,true)*Fermi(L/2-1,L,false)*Fermi(L/2,L,true)*Fermi(L/2,L,false)*U;
    return h.toMPS().Sweep();
}


//------------------------ Measurements -------

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


void ExportSTable(string filename,const stdvec& qs,const TensorD& s)
{
    ofstream out(filename.c_str());
    out<<"r ";
    for(auto q:qs) out<<"q="<<q<<" ";
    out<<endl;
    for(int i=0;i<s.dim[0];i++)
    {
        out<<i<<" ";
        for(int j=0;j<int(qs.size());j++)
            out<<s[{i,j}]<<" ";
        out<<endl;
    }
}

void CalculateS()
{
    MPS gs;
    gs.Load("gs.dat");
    vector<double> qs={0.5, 1, 1.5, 2, 5, 10};
    TensorD s({ gs.length-1, int(qs.size()) });
    for(int i=0;i<gs.length-1;i++)
    {
        gs.SetPos({i,1});
        TensorD rho=gs.C*gs.C.t();
        TensorD eval=rho.EigenDecomposition(1).at(1);
        for(int j=0;j<int(qs.size());j++)
            s[{i,j}]=EntropyRenyi(eval.data(),eval.size(),qs[j]);
    }

    ExportSTable("entropy.dat",qs,s);
}

//---------------------------- Test DMRG basico -------------------------------------------

void TestDMRGBasico(const Parameters &par)
{
    int len=par.length;
    auto op=HamSiamTB(len,1,1);
//    auto op=HamSiamTBStar(len,1,1);
    op.Sweep(); op.PrintSizes("HamMPO=");
    op.decomposer=MatQRDecomp;
    auto nop=NParticle(len);

    DMRG1_wse_gs sol(op,par.m);
    sol.tol_diag=1e-4;
    for(int k=0;k<par.nsweep;k++)
    {
        std::cout<<"sweep "<<k+1<<" --------------------------------------\n";
        for(auto p : MPS::SweepPosSec(op.length))
        {
            sol.SetPos(p);
            sol.Solve();
            sol.Print();
        }
        cout<<"nT="<<Superblock({&sol.gs,&nop,&sol.gs}).value()<<endl;
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
        CalculateS();
    }
    cout<<"\nDone in "<<difftime(time(NULL),t0)<<"s"<<endl;
    return 0;
}
