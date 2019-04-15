#ifndef SUPERBLOCK_H
#define SUPERBLOCK_H

#include"mps.h"
#include<vector>
#include<array>

#include"supertensor.h"

class Superblock
{
    TensorD one;
 public:
    std::vector<MPS*> mps;
    int length;
    MPS::Pos pos={0,-1};
    std::vector<TensorD> b1,b2;

    Superblock() {}
    Superblock(const std::vector<MPS*>& mps)
        :mps(mps),length(mps[0]->length),b1(length),b2(length)
    {
        one=TensorD(Index(mps.size(),1),{1});
        InitBlocks();
    }
    void InitBlocks()
    {
        pos=mps.front()->pos;
        for(MPS* x:mps)
            x->SetPos(pos);
        auto prod=one;
        for(int i=0;i<=pos.i;i++)
            b1[i]=prod=prod*Transfer(i);
        prod=one;
        for(int i=length-2;i>=pos.i;i--)
            b2[i]=prod=Transfer(i+1)*prod;
    }
    double norm_factor() const
    {
        double prod=1;
        for(const MPS* x:mps) prod*=x->norm_factor();
        return prod;
    }
    SuperTensor Oper() const
    {
        if(mps.size()==3)
            return { b1[pos.i]*norm_factor(), b2[pos.i], {mps[1]->C} };
        else return { b1[pos.i]*norm_factor(), b2[pos.i]};
    }
    double value() const
    {
        TensorD Cp=Oper()*mps.front()->C;
        return Dot(mps.back()->C,Cp);
    }
    void SetPos(MPS::Pos p)
    {
        if (pos.i<0 || pos.i>length-2)
            throw std::invalid_argument("SB::SetPos out of range");
        while(pos<p) SweepRight();
        while(pos>p) SweepLeft();
    }
    void SweepRight()
    {
        if(pos.i==length-2 && pos.vx==1) return;
        ++pos;
        for(MPS* x:mps) x->SetPos(pos);
//        for(MPS* x:mps) x->SweepRight();
//        pos=mps.front()->pos;
//        vx=mps.front()->vx;
        if (pos.i==0)
            b1[pos.i]=one*Transfer(pos.i);
        else
            b1[pos.i]=b1[pos.i-1]*Transfer(pos.i);
    }
    void SweepLeft()
    {
        if (pos.i==0 && pos.vx==-1) return;
        --pos;
        for(MPS* x:mps) x->SetPos(pos);
//        for(MPS* x:mps) x->SweepLeft();
//        pos=mps.front()->pos;
//        vx=mps.front()->vx;
        if (pos.i==length-2)
            b2[pos.i]=Transfer(pos.i+1)*one;
        else
            b2[pos.i]=Transfer(pos.i+1)*b2[pos.i+1];
    }
    std::vector<const TensorD*> Transfer(int pos) const
    {
        std::vector<const TensorD*> transfer;
        for(const MPS* x:mps)
            transfer.push_back( &x->at(pos) );
        return transfer;
    }

};

#endif // SUPERBLOCK_H
