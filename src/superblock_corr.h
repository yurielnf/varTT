#ifndef SUPERBLOCK_CORR_H
#define SUPERBLOCK_CORR_H

#include "mps.h"

class SuperBlock_Corr
{
    MPS& gs;
    TensorD left;
    int i0=-1, j0=-1;
public:
    SuperBlock_Corr(MPS& gs_)
        : gs(gs_)
    {}

    void computeLeft(const MPO& mpo,int i, int j)
    {
        if (i==gs.length-1)
            gs.SetPos({i-1,-1});
        else
            gs.SetPos({i,1});
        left=TensorD({gs.at(i).dim.front(), 1, gs.at(i).dim.front()});
        left.FillEye(2);

        for(auto p=i; p<j; p++)
            left= left * Transfer(mpo,p);
    }

    void updateLeft(const MPO& mpo,int j)
    {
        for(auto p=j0; p<j; p++)
            left= left * Transfer(mpo,p);
    }

    double value(const MPO& mpo, int i, int j)
    {
        if (i0!=i)
            computeLeft(mpo,i,j);
        else
            updateLeft(mpo,j);
        i0=i;
        j0=j;
        auto oneR=TensorD({gs.at(j).dim.back(), 1, gs.at(j).dim.back()});
        oneR.FillEye(2);
        return Dot(left*Transfer(mpo,j), oneR)
                * pow(gs.norm_factor(),2)
                * mpo.norm_factor();
    }

 private:
    mutable TensorD Mgs, Mo;

    std::vector<const TensorD*> Transfer(const MPO& mpo,int i) const
    {
        Mgs= (gs.pos==MPS::Pos{i,1}) ? gs.at(i)*gs.C :
                                       (gs.pos==MPS::Pos{i-1,-1}) ? gs.C*gs.at(i)
                                                                : gs.at(i);
        Mo=(i==mpo.pos.i) ? mpo.at(i)*mpo.C : mpo.at(i);
        return {&Mgs, &Mo, &Mgs};
    }
};

#endif // SUPERBLOCK_CORR_H
