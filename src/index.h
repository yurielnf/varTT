#ifndef INDEX_H
#define INDEX_H

#include<vector>
#include<array>
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
Index ToIndex(int pos, Index dim);

std::vector<Index> SplitIndex(Index dim,int splitPos);
std::vector<Index> SplitIndex(Index dim,std::vector<int> splitPos);

Index IndexReorder(const Index& dim, const std::vector<int>& posMap);

Index IndexMul(const Index& dim1,const Index& dim2); // Dim resulting from matrix multiplication

//-------------------------------------- string manipulation ------------------------------

bool ArePermutation(std::string str1,std::string str2);
std::vector<int> Permutation(std::string str1,std::string str2);
std::array<std::string,2> SortForMultiply(std::string str1,std::string str2);


#endif // INDEX_H
