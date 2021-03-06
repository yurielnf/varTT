#ifndef PARAMETERS_H
#define PARAMETERS_H
#include<string>
#include<vector>

struct Parameters
{
    int length=10,
        nsweep=4, nsweep_resid=2, nsweep_jd=1,
        m=128,
        nkrylov=2,
        nsite_gs=0, nsite_resid=1, nsite_jd=0,
        spin=1,
        periodic=0,
        useLanczosCv=1;
    char opType='C';
    int op1Pos=0,
         op2Pos=0;
    std::string opProj="";
    double etaFactor=1,
           DSz2=0, BF=0;
    std::string hallW_file="W.dat";
    std::vector<double> renyi_q;
    int freeFermionLx=100;
    double mu=0;
    Parameters(){}
    Parameters(const char file_name[]) {ReadParameters(file_name);}

    void ReadParameters(const char filename[]);
};


#endif // PARAMETERS_H
