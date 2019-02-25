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
    TensorD C;

    Superblock(const std::vector<MPS>& mps)
        :mps(mps),length(mps[0].length),b1(length),b2(length)
    {
        C=TensorD(Index(mps.size(),1),{1});
        Canonicalize();
    }
    void Canonicalize()
    {
        SetPos(length-1);
        SetPos(-1);
        SetPos(length/2-1);
    }
    void SetPos(int p)
    {
        while(pos<p) SweepRight();
        while(pos>p) SweepLeft();
    }
    double value() const
    {
        auto prod=mps[0].C("iI") * b1[pos]("ijk") * mps[2].C("kK") * b2[pos+1]("IJK") * mps[1].C("jJ");
        return prod.t[0]*mps[1].norm();
    }
    void SweepRight()
    {
        if(pos==length-1) return;
        pos++;
        for(MPS& x:mps) x.SetPos(pos);
        if (pos==0)
            b1[pos]=C*Transfer();
        else
            b1[pos]=b1[pos-1]*Transfer();
    }
    void SweepLeft()
    {
        if (pos<0) return;
        for(MPS& x:mps) x.SetPos(pos-1);
        if (pos==length-1)
            b2[pos]=Transfer()*C;
        else
            b2[pos]=Transfer()*b2[pos+1];
        pos--;
    }
    std::vector<const TensorD*> Transfer() const
    {
        std::vector<const TensorD*> transfer;
        for(const MPS& x:mps)
            transfer.push_back( &x.at(pos) );
        return transfer;
    }    

};

#endif // SUPERBLOCK_H
