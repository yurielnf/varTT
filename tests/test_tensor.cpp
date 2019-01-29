#include"tensor.h"
#include<iostream>
#include"catch.hpp"

using namespace std;

TEST_CASE( "tensor level 1", "[tensor]" )
{
    Tensor<double> t({2,3,2});
    SECTION( "size" )
    {
        REQUIRE( t.size() == 2*3*2 );
    }
    SECTION( "fill" )
    {
        t.FillZeros();
        REQUIRE( t[{1,1,1}]==0 );
        t.FillRandu();
        REQUIRE( t[{1,1,1}]!=0 );
    }
    SECTION( "assign" )
    {
        t[{1,1,1}]=43;
        REQUIRE( t[{1,1,1}]==43 );
    }
    SECTION( "save/load" )
    {
        auto &t1=t;
        t1.FillRandu();
        t1.Save("t1.txt");
        Tensor<double> t2(t1.dim);
        t2.Load("t1.txt");
        t2.Save("t2.txt");
        for(int i=0;i<t1.size();i++)
            REQUIRE( t1.data[i]==t2.data[i] );
    }
    SECTION( "operator-/Norm" )
    {
        auto dt=t-t;
        REQUIRE( Norm(dt)<1e-16 );
    }
}
