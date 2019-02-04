#include"index.h"

using namespace std;

int Offset(Index id, Index dim)
{
    int sum=0,prod=1;
    for(uint i=0;i<id.size();i++)
    {
        sum+=id[i]*prod;
        prod*=dim[i];
    }
    return sum;
}

vector<Index> SplitIndex(Index dim,int splitPos)
{
    vector<Index> dim_v(2);
    for(int i=0;i<splitPos;i++)
        dim_v[0].push_back(dim[i]);
    for(int i=splitPos;i<dim.size();i++)
        dim_v[1].push_back(dim[i]);
    for(auto& x:dim_v)
        if (x.empty()) x.push_back({1});
    return dim_v;
}

vector<Index> SplitIndex(Index dim,vector<int> splitPos)
{
    vector<Index> dim_v(splitPos.size()+1);
    uint p=0,s;
    for(s=0;s<splitPos.size();s++)
    {
        for(int i=p;i<splitPos[s];i++)
            dim_v[s].push_back(dim[i]);
        p=splitPos[s];
    }
    for(uint i=p;i<dim.size();i++)
        dim_v[s].push_back(dim[i]);
    if (dim_v.front().empty()) dim_v.front().push_back({1});
    if (dim_v.back() .empty()) dim_v.back() .push_back({1});
    return dim_v;
}

Index IndexMul(const Index& dim1,const Index& dim2) // Dim resulting from matrix multiplication
{
    if (dim1.back()!=dim1.front())
        throw std::invalid_argument("Index:: IndexMul() incompatible dimensions");

    Index dim_r;
    for(int i=0;i<dim1.size()-1;i++)
        dim_r.push_back( dim1[i] );
    for(int i=1;i<dim2.size();i++)
        dim_r.push_back( dim2[i] );
    return dim_r;
}
