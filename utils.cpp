#include "utils.h"
#include<algorithm>

using namespace std;

int Prod(std::vector<int> dim)
{
    return accumulate(dim.begin(),dim.end(),1,multiplies<int>());
}

int Offset(std::vector<int> id, std::vector<int> dim)
{
    int sum=0,prod=1;
    for(uint i=0;i<id.size();i++)
    {
        sum+=id[i]*prod;
        prod*=dim[i];
    }
    return sum;
}
