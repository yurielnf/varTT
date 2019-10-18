#ifndef HAMHALL_H
#define HAMHALL_H

#include<armadillo>
#include"mps.h"

using namespace std;

class HamHall
{

public:
    int length,d_cut_exp=10,d_cut_Fourier=10;
    double tol=1e-13;
    std::string in_W="W.dat";

    vector<double> ruido;
    arma::mat W;
    bool periodic=false;
    double noiseAmp=0, mu=0;

    HamHall(int length)
        :length(length)
        ,ruido(length,0.0)
    {}

    void Load()
    {
        if (!W.load(in_W,arma::raw_ascii))
            throw std::invalid_argument("W.dat file not found");
        W=W.t();
        cout<<"size="<<W.n_rows<<"x"<<W.n_cols<<endl;
        if (periodic) W = Periodic(W,length);

        for(int jj=0;jj<length;jj++)
        {
            ruido[jj]=mu;
            for(int ll=0;ll<length;ll++)
                ruido[jj]+=-W(ToMatrixIndex(jj-ll),ToMatrixIndex(0))
                           +W(ToMatrixIndex(0),ToMatrixIndex(ll-jj));
//            cout<<ruido[jj]<<endl;
        }
        if (noiseAmp==0) return;
        std::random_device rd{};
        std::mt19937 gen{rd()};
        normal_distribution<double> d{0,noiseAmp};
        for(double &x:ruido) x+=d(gen);
    }

    arma::mat Periodic(const arma::mat& W,int L) const
    {
        int n=(W.n_rows+1)/(2*L)-1;
        arma::mat Wnew(2*L-1,2*L-1);
        Wnew.fill(0.0);
        for(int i=-L+1;i<L;i++)
            for(int j=-L+1;j<L;j++)
                for(int a=-n;a<=n;a++)
                    for(int b=-n;b<=n;b++)
                        Wnew(i+L-1,j+L-1)+=W(ToMatrixIndex(i+a*L),
                                             ToMatrixIndex(j+b*L));
        return Wnew;
    }

    int ToMatrixIndex(int i) const
    {
        int len=(W.n_rows+1)/2;
        return i+len-1;
    }

    MPO KineticEnergy() const
    {
        MPSSum T(10,MatSVDFixedTol(tol));
        for(int i=0;i<length;i++)
            if (fabs(ruido[i])>1e-5)
                T+=Fermi(i,length,true)*Fermi(i,length,false)*ruido[i];
        return T.toMPS();
    }

    int SiteDist(int i,int j) const
    {
        int d1=abs(i-j);
        int d2=abs(length-d1);
        if (periodic) return min(d1,d2);
        else return d1;
    }

    int ToSite(int x) const
    {
        if (periodic) return (x+length)%length;
        if (x<0 || x>=length) return -1; //invalid
        else return x;
    }

    MPO InteractionU3() const
    {
        MPSSum pot=MPSSum(10,MatSVDFixedTol(tol));
//        for(int jj=0; jj<length; jj++)
        for(int jj=length-1; jj>=0; jj--)
        {
            for(int mm=0; mm < length;mm++)
            {
                if (SiteDist(mm,jj)>d_cut_exp) continue;
                for(int kk=mm+1 ; kk < length;kk++)
                {
                    if (SiteDist(kk,jj)>d_cut_Fourier) continue;
                    for(int ll=jj+1; ll < length;ll++)
                        if(( (!periodic && kk+mm==jj+ll )) ||
                                (periodic && (kk+mm)%length==(jj+ll)%length))
                        {
                            double coef= W( ToMatrixIndex(kk-jj),ToMatrixIndex(mm-jj))
                                    -W( ToMatrixIndex(kk-ll),ToMatrixIndex(mm-ll))
                                    -W( ToMatrixIndex(mm-jj),ToMatrixIndex(kk-jj))
                                    +W( ToMatrixIndex(mm-ll),ToMatrixIndex(kk-ll));
                            if (fabs(coef)>tol)
                            {
                                auto term=Fermi(jj,length,true )*Fermi(ll,length,true)*
                                          Fermi(kk,length,false)*Fermi(mm,length,false);
                                pot+=term*coef;
                            }
                        }
                }
            }
        }
        return pot.toMPS().Sweep();
    }

    MPO Hamiltonian() const
    {
        return KineticEnergy()+InteractionU3();
//        MPSSum hs(10,MatSVDFixedTol(tol));
//        hs+=KineticEnergy();
//        hs+=InteractionU3();
//        MPO h=hs.toMPS();
//        hs+=h*ElectronHoleMPO(length);
//        return hs.toMPS()*0.5;
    }
};


#endif // HAMHALL_H
