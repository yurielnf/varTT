#ifndef SUPERBLOCK_H
#define SUPERBLOCK_H

#include"mps.h"
#include<vector>
#include<array>

class Superblock
{
 public:
    std::vector<MPS> mps;
    int length,pos=-1;
    std::vector<TensorD> b1,b2;

    Superblock(const std::vector<MPS>& mps)
        :mps(mps),length(mps[0].length),b1(length),b2(length)
    {}

    void Canonicalize()
    {
        SetPos(-1);
        while(pos<length/2-1)
            SweepRight();
        SetPos(length-1);
        while(pos>length/2-1)
            SweepLeft();
    }
    double value() const
    {
        auto prod=mps[0].C("iI") * b1[pos]("ijk") * mps[2].C("kK") * b2[pos+1]("IJK") * mps[1].C("jJ");
        return prod.t[0]*mps[1].norm();
    }

    void SweepRight()
    {
        SetPos(pos+1);
        if (pos==0)
        {
            Index dimC(mps.size(),1);
            TensorD C(dimC,{1});
            b1[pos]=C*Transfer(false);
        }
        else
            b1[pos]=b1[pos-1]*Transfer(false);
    }
    void SweepLeft()
    {
        SetPos(pos-1);
        if (pos==length-2)
        {
            Index dimC(mps.size(),1);
            TensorD C(dimC,{1});
            b2[pos+1]=Transfer(true)*C;
        }
        else
            b2[pos+1]=Transfer(true)*b2[pos+1];
    }
    std::vector<const TensorD*> Transfer(bool isB) const
    {
        std::vector<const TensorD*> transfer;
        for(const MPS& x:mps) transfer.push_back(&x.M[pos+isB]);
        return transfer;
    }
    void SetPos(int p)
    {
        pos=p;
        for(MPS& x:mps) x.SetPos(p);
    }

};

#endif // SUPERBLOCK_H
