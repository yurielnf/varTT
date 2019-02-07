#include "tensor.h"
#include"catch.hpp"

#include<string>

using namespace std;

TEST_CASE( "tensor notation", "[tnotation]" )
{
    SECTION( "Indices are permutation" )
    {
        REQUIRE(  ArePermutation("ijkl","ilkj") );
        REQUIRE( !ArePermutation("ijkl","ilj") );
        REQUIRE( !ArePermutation("ijkl","iljki") );
        REQUIRE(  ArePermutation("ijkl","lkji") );
        REQUIRE( !ArePermutation("ijkl","lkjig") );
    }
    SECTION( "Indices permutation to vector of pos" )
    {
        REQUIRE( Permutation("ijkl","ilkj")==vector<int>({0,3,2,1}) );
        REQUIRE( Permutation("ijkl","lkji")==vector<int>({3,2,1,0}) );
        REQUIRE( Permutation("ikl","lki")==vector<int>({2,1,0}) );
        REQUIRE( Permutation("ijk","jki")==vector<int>({2,0,1}) );
    }
    SECTION( "matrix transpose" )
    {
        TensorD t1({2,3,2}), t2;
        t1.FillRandu();
        t2("jki")=t1("ijk");
        REQUIRE( t2==t1.Transpose(1));
    }

}
