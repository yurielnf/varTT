#ifndef PARAMETERS_H
#define PARAMETERS_H


struct Parameters
{
    int length=10,
        nsweeps=4,
        m=128;
    char opType='C';
    int op1Pos=0,
         op2Pos=0;
    double etaFactor=1;

    void ReadParameters(char filename[]);
};


#endif // PARAMETERS_H
