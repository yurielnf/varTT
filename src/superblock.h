#ifndef SUPERBLOCK_H
#define SUPERBLOCK_H

#include"mps.h"
#include<vector>
#include<array>

#include"supertensor.h"

class Superblock
{
 public:
    std::vector<MPS> mps;
    int length,pos=0;
    std::vector<TensorD> b1,b2;
    TensorD one;

    Superblock(const std::vector<MPS>& mps)
        :mps(mps),length(mps[0].length),b1(length),b2(length)
    {
        one=TensorD(Index(mps.size(),1),{1});
        InitBlocks();
    }
    void InitBlocks()
    {
        b1[0]=one*Transfer(pos);
        b2[length-2]=Transfer(length-1)*one;
        pos=0;
        for(MPS& x:this->mps)
            x.SetPos(pos);
        for(int k=0;k<1;k++)
        for(int i:MPS::SweepPosSec(length)) SetPos(i);
    }
    double Norm() const
    {
        double prod=1;
        for(const MPS& x:mps) prod*=x.norm_factor();
        return prod;
    }
    SuperTensor Oper() const
    {
        return { b1[pos], mps[1].C, b2[pos] };
    }
    double value() const
    {
        TensorD Cp=Oper()*mps[0].C;
        auto value=Cp("kK")*mps[2].C("kK");
//        auto value=mps[0].C("iI") * b1[pos]("ijk") * mps[2].C("kK") *
//                   b2[pos]("IJK") * mps[1].C("jJ") ;

        return value.t[0]*Norm();
    }

    void SetPos(int p)
    {
        if (pos<0 || pos>length-2)
            throw std::invalid_argument("SB::SetPos out of range");
        while(pos<p) SweepRight();
        while(pos>p) SweepLeft();
    }

    void SweepRight()
    {
        if(pos==length-2) return;
        for(MPS& x:mps) x.SweepRight();
        pos=mps.front().pos;
        if (pos==0)
            b1[pos]=one*Transfer(pos);
        else
            b1[pos]=b1[pos-1]*Transfer(pos);
    }
    void SweepLeft()
    {
        if (pos==0) return;
        for(MPS& x:mps) x.SweepLeft();
        pos=mps.front().pos;
        if (pos==length-2)
            b2[pos]=Transfer(pos+1)*one;
        else
            b2[pos]=Transfer(pos+1)*b2[pos+1];
    }
    std::vector<const TensorD*> Transfer(int pos) const
    {
        std::vector<const TensorD*> transfer;
        for(const MPS& x:mps)
            transfer.push_back( &x.at(pos) );
        return transfer;
    }

};

#endif // SUPERBLOCK_H
