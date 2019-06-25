#include"parameters.h"
#include<fstream>
#include<iostream>
#include<string>
#include<stdexcept>

using namespace std;

void Parameters::ReadParameters(char filename[])
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
       else if (param=="nsweeps")
            in>>nsweeps;
        else if (param=="m")
            in>>m;
        else if(param=="opType")
            in>>opType;
        else if(param=="op1Pos")
            in>>op1Pos;
        else if(param=="op2Pos")
            in>>op2Pos;
        else if(param=="etaFactor")
            in>>etaFactor;
        in.ignore(1000,'\n');
    }
}
