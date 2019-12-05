#include"parameters.h"
#include<fstream>
#include<iostream>
#include<string>
#include<stdexcept>
#include<sstream>

using namespace std;

void Parameters::ReadParameters(const char filename[])
{
    ifstream in(filename);
    if (!in.is_open())
        throw std::invalid_argument("I couldn't open parameter file");
    string param;
    while (!in.eof())
    {
        in>>param;
        if (param=="length")
            in>>length;
       else if (param=="nsweep")
            in>>nsweep;
        else if (param=="nsweep_resid")
             in>>nsweep_resid;
        else if (param=="nsweep_jd")
             in>>nsweep_jd;
        else if (param=="m")
            in>>m;
        else if (param=="nkrylov")
            in>>nkrylov;
        else if (param=="nsite_gs")
            in>>nsite_gs;
        else if (param=="nsite_resid")
            in>>nsite_resid;
        else if (param=="nsite_jd")
            in>>nsite_jd;
        else if (param=="spin")
            in>>spin;
        else if (param=="periodic")
            in>>periodic;
        else if(param=="opType")
            in>>opType;
        else if(param=="op1Pos")
            in>>op1Pos;
        else if(param=="op2Pos")
            in>>op2Pos;
        else if(param=="etaFactor")
            in>>etaFactor;
        else if(param=="DSz2")
            in>>DSz2;
        else if(param=="BF")
            in>>BF;
        else if(param=="hallW_file")
            in>>hallW_file;
        else if(param=="renyi_q")
        {
            getline(in,param);
            istringstream iss(param);
            string word;
            while(iss >> word)
                renyi_q.push_back(stod(word));
        }
        else if(param=="freeFermionLx")
            in>>freeFermionLx;
        else if(param=="muHall")
            in>>muHall;
        in.ignore(1000,'\n');
    }
}
