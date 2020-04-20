#ifndef HAMNN_H
#define HAMNN_H

#include<armadillo>
#include<mps.h>
#include<fstream>
#include<map>
#include<array>

class HamNN
{
public:
    int L;
    std::vector<int> impPos;
    arma::mat kin, potM;
    std::map<std::array<int,4>,double> Vijkl;

    std::string
        file_impPos="impPos.dat",
        file_kin="kin.dat",
        file_potM="potM.dat",
        in_Vijkl="Vijkl.dat";

    HamNN(int len):L(len){}
    void Load()
    {
        {
            std::ifstream in(file_impPos);
            if (!in.is_open())
                throw std::invalid_argument("HamiltonianNN: impPosFile not found");
            double x;
            in>>x;
            while( !in.eof()) { impPos.push_back(x);in>>x;  }
        }
        kin.load(file_kin,arma::arma_ascii);
        if (kin.n_rows != L)
            throw std::invalid_argument("HamiltonianNN: kinFile isn't compatible with this length");
        potM.load(file_potM,arma::arma_ascii) ;
        if (potM.n_rows != L)
            throw std::invalid_argument("HamiltonianNN: potFile isn't compatible with this length");

        if (in_Vijkl!="")
        {
            std::ifstream in(in_Vijkl);
            if (!in.is_open())
                throw std::invalid_argument("HamiltonianNN: Vijkl file not found");
            int i,j,k,l;
            double x;
            in>>i>>j>>k>>l>>x;
            while( !in.eof()) {
                array<int,4> ind={i,j,k,l};
                if (fabs(x)>1e-12) Vijkl[ind]=x;
                in>>i>>j>>k>>l>>x;
            }
        }
    }

    MPO KineticEnergy() const
    {
        const int m=1;
        MPSSum h(m,MatSVDFixedTol(1e-13));
        for(int i=0;i<L;i++)
            for(int j=0;j<L;j++)
                if (fabs(kin(i,j))>1e-12)
                    h += Fermi(i,L,true)*Fermi(j,L,false)*kin(i,j);
        return h.toMPS();
    }
    MPO InteractionU() const
    {
        const int m=1;
        MPSSum h(m,MatSVDFixedTol(1e-13));
        for(int i=0;i<L;i++)
            for(int j=0;j<L;j++)
                if (fabs(potM(i,j))>1e-12)
                    h += Fermi(i,L,true)*Fermi(i,L,false)*
                         Fermi(j,L,true)*Fermi(j,L,false)*potM(i,j);
        for(const auto& x:Vijkl)
        {
            const auto& ar=x.first;
            const auto& value=x.second;
            h += Fermi(ar[0],L,true)*Fermi(ar[1],L,true)*
                 Fermi(ar[2],L,false)*Fermi(ar[3],L,false)*value;
        }


        return h.toMPS();
    }
    MPO toMPO() const
    {
        return KineticEnergy()+InteractionU();
    }
};



#endif // HAMNN_H
