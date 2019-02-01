#ifndef INDEX_H
#define INDEX_H

#include<vector>
#include<algorithm>
#include<stdexcept>

template<class T>
int Prod_n(const T* ini, int n)
{
    return std::accumulate(ini,ini+n,1,std::multiplies<T>());
}

template<class T>
int Prod(const T* ini, const T* fin)
{
    return std::accumulate(ini,fin,1,std::multiplies<T>());
}

typedef std::vector<int> Index;

inline int Prod(std::vector<int> dim)
{
    return Prod_n(&dim[0],dim.size());
}

int Offset(Index id, Index dim);

std::vector<Index> SplitIndex(Index dim,int splitPos);
std::vector<Index> SplitIndex(Index dim,std::vector<int> splitPos);

Index IndexMul(const Index& dim1,const Index& dim2); // Dim resulting from matrix multiplication

#endif // INDEX_H
