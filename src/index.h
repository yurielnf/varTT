#ifndef INDEX_H
#define INDEX_H

#include<vector>
#include<array>
#include<algorithm>
#include<stdexcept>

template<class T>
T Prod_n(const T* ini, int n)
{
    return std::accumulate(ini,ini+n,1,std::multiplies<T>());
}

template<class T>
T Prod(const T* ini, const T* fin)
{
    return std::accumulate(ini,fin,1,std::multiplies<T>());
}

typedef std::vector<int> Index;

inline int Prod(const std::vector<int>& dim)
{
    return Prod_n(&dim[0],dim.size());
}

int Offset(const Index& id, const Index& dim);
int Offset(const Index &id, const Index &dim, const std::vector<int> &posMap);
Index ToIndex(int pos, const Index& dim);

std::vector<Index> SplitIndex(const Index& dim,int splitPos);
std::vector<Index> SplitIndex(const Index& dim,
                              const std::vector<int>& splitPos);

Index IndexReorder(const Index& dim, const std::vector<int>& posMap);

Index IndexMul(const Index& dim1,const Index& dim2,int nIndCommon); // Dim resulting from matrix multiplication

//-------------------------------------- string manipulation ------------------------------

bool ArePermutation(std::string str1,std::string str2);
std::vector<int> Permutation(std::string str1,std::string str2);
std::array<std::string,4> SortForMultiply(std::string str1,std::string str2);


#endif // INDEX_H
