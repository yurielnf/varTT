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
Index ToIndex(int pos,Index dim)
{
    Index id(dim.size());
    for(int i=0;i<dim.size();i++)
    {
        id[i]=pos%dim[i];
        pos/=dim[i];
    }
    return id;
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

Index IndexReorder(const Index& dim, const std::vector<int> &posMap)
{
    Index dim2(dim.size());
    for(int i=0;i<dim.size();i++)
        dim2[posMap[i]]=dim[i];
    return dim2;
}

Index IndexMul(const Index& dim1,const Index& dim2) // Dim resulting from matrix multiplication
{
    if (dim1.back()!=dim2.front())
        throw std::invalid_argument("Index:: IndexMul() incompatible dimensions");

    Index dim_r;
    for(int i=0;i<dim1.size()-1;i++)
        dim_r.push_back( dim1[i] );
    for(int i=1;i<dim2.size();i++)
        dim_r.push_back( dim2[i] );
    return dim_r;
}

//-------------------------------------------- string manipulation ------------------------------

bool ArePermutation(std::string str1,std::string str2)
{
    auto s1=str1; sort(s1.begin(),s1.end());
    auto s2=str2; sort(s2.begin(),s2.end());
    return s1==s2;
}
std::vector<int> Permutation(std::string ini,std::string fin)
{
    if (ini==fin) return {};
    if (!ArePermutation(ini,fin))
        throw std::invalid_argument("Permutation: str1,str2 is not a permutation");
    std::vector<int> pos(ini.size());
    for(uint i=0;i<ini.size();i++)
        pos[i]=fin.find(ini[i]);
    return pos;
}
std::array<std::string,3> SortForMultiply(std::string str1,std::string str2)
{
    auto s1=str1; sort(s1.begin(),s1.end());
    auto s2=str2; sort(s2.begin(),s2.end());
    std::string sc;
    std::set_intersection(s1.begin(),s1.end(),
                          s2.begin(),s2.end(),
                          std::back_inserter(sc));
    s1.clear();
    for(int i=0;i<str1.size();i++)
        if (sc.find(str1[i])==std::string::npos)
            s1.push_back(str1[i]);

    s2.clear();
    for(int i=0;i<str2.size();i++)
        if (sc.find(str2[i])==std::string::npos)
            s2.push_back(str2[i]);
    return {s1+sc,sc+s2,s1+s2};
}
