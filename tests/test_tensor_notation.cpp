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
    TensorD t1({2,3,2}), t2;
    t1.FillRandu();
    SECTION( "matrix transpose" )
    {
        t2("nml")=t1("mln");            //t2=t1.Reorder("mln","nml");
        REQUIRE( t2==t1.Transpose(2));
    }
    SECTION( "matrix multiplication" )
    {
        t2("ijlm")=t1("ijk") * t1("klm");

        REQUIRE( t2.dim==Index{2,3,3,2} );
        REQUIRE( t2.vec()==(t1*t1).vec() );
    }
    SECTION( "contraction and reorder" )
    {
        t2("lmij")=t1("ijk") * t1("klm");

        REQUIRE( t2.dim==Index{3,2,2,3} );
        REQUIRE( t2.vec()==(t1*t1).Transpose(2).vec() );

        t2("mlij")=t1("ijk") * t1("mlk");

        REQUIRE( t2.dim==Index{2,3,2,3} );
        REQUIRE( t2.vec()==(t1*t1.Transpose(2)).Transpose(2).vec() );

        t2("ikml")=t1("ijk") * t1("ljm");

        REQUIRE( t2.dim==Index{2,2,2,2} );
    }

}
