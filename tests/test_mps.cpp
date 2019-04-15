#include "mps.h"
#include"catch.hpp"


TEST_CASE( "mps canonization", "[mps]" )
{
    MPS x(20,8);
    x.FillRandu({8,2,8});
    SECTION( "initialization and compatibility" )
    {
        for(int i=0;i<2;i++)
        {
            x.PrintSizes();
            int m=1;
            for(int i=0;i<x.length;i++)
            {
                auto t=x.at(i);
                REQUIRE( t.dim[0]==m );
                REQUIRE( t.dim[1]==2 );
                REQUIRE( t.dim[2]<=x.m );
                m=t.dim[2];
            }            
        }
    }
    SECTION( "Sweep, casting as tensor")
    {
        MPS x(10,8,MatChopDecompFixedTol(0));
        x.FillRandu({8,2,8});
        TensorD tx=x;
        while (x.pos.i<x.length-2)
        {
            x.SweepRight();
            TensorD tx2=x;
            REQUIRE( Norm(tx-tx2)/Norm(tx) < 1e-14 );
        }
        while (x.pos.i>0)
        {
            x.SweepLeft();
            TensorD tx2=x;
            REQUIRE( Norm(tx-tx2)/Norm(tx) < 1e-14 );
        }
        SECTION( "SetPos")
        {
            x.SetPos({x.length/2,1});
            TensorD tx2=x;
            REQUIRE( Norm(tx-tx2)/Norm(tx) < 1e-14 );
        }
        SECTION( "Canonicalize")
        {
            x.Canonicalize();
            TensorD tx2=x;
            REQUIRE( Norm(tx-tx2)/Norm(tx) < 1e-13 );
        }
    }
    SECTION( "norm" )
    {
        MPS x(20,8);
        x.FillRandu({8,2,8});
        x.Canonicalize();
        x.Normalize();
        REQUIRE( x.norm() ==Approx(1) );
        REQUIRE( Norm(x.C)==Approx(1) );
    }
    SECTION( "MPS operators: * +" )
    {
        x.Canonicalize(); x.Normalize();
        REQUIRE( x.norm()==Approx(1) );
        REQUIRE( (x*3).norm()==Approx(3) );
        x+=x;
        REQUIRE( x.norm()==Approx(2) );
        auto dif=x;
        dif+=x*(-1.0);
        dif.Canonicalize();
        REQUIRE( dif.norm() < 1e-13 );
    }

}
