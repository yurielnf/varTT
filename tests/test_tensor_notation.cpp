#include "tensor.h"
#include"catch.hpp"

#include<string>

using namespace std;

TEST_CASE( "tensor notation", "[tnotation]" )
{
    SECTION( "Indices are permutation" )
    {
        REQUIRE( ArePermutation("ijkl","ilkj")==true );
        REQUIRE( ArePermutation("ijkl","ilj")==false );
        REQUIRE( ArePermutation("ijkl","iljki")==false );
        REQUIRE( ArePermutation("ijkl","lkji")==true );
        REQUIRE( ArePermutation("ijkl","lkjig")==false );
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
        t2("nml")=t1("mln");            //t2=t1.Reorder("mln","nml");
        REQUIRE( t2==t1.Transpose(2));
    }

}
