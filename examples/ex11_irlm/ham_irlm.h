#ifndef HAMHALL_H
#define HAMHALL_H

#include<armadillo>
#include"mps.h"

using namespace std;

class HamIRLM
{

public:
    arma::mat tmat, Pmat;
    double U;
    double tol=1e-10;

    HamIRLM(string tFile,string PFile, double U_)
        :U(U_)
    {
        tmat.load(tFile);
        Pmat.load(PFile);

        cout<<"P*P.t()-1 = "<< arma::norm(Pmat*Pmat.t()-arma::mat(length(),length(), arma::fill::eye)) << endl;
    }

    int length() const { return tmat.n_rows; }
    MPO Create(int i) const { return Fermi(i,length(),true); }
    MPO Destroy(int i) const { return Fermi(i,length(),false); }

    MPO Ham() const
    {
        cout<<"Building Hamiltonian MPO ...\n";
        int L=length();
        auto h=MPSSum(10,MatSVDFixedTol(tol));;
        // kinetic energy bath
        for(int i=0;i<L; i++)
            for(int j=0;j<L; j++)
        {
            if (fabs(tmat(i,j))<tol) continue;
            h += Create(i)*Destroy(j)*tmat(i,j) ;
        }

        //interaction
        for(int a=0;a<L; a++) { cout<<a<<" "; cout.flush();
            for(int b=0;b<L; b++)
                for(int c=0;c<L; c++)
                    for(int d=0;d<L; d++) {
                        double coeff=U*Pmat(0,a)*Pmat(1,b)*Pmat(1,c)*Pmat(0,d);
                        if (c==d || a==b || fabs(coeff)<1e-5) continue;
                        h += Create(a)*Create(b)*Destroy(c)*Destroy(d)* coeff;
                    }
        }
        cout<<"Done sum\n"; cout.flush();
        auto H=h.toMPS().Sweep();
        cout<<"Done Ham\n"; cout.flush();
        return H;
    }
};


#endif // HAMHALL_H
