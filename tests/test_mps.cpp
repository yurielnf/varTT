#include "mps.h"
#include"catch.hpp"


TEST_CASE( "mps canonization", "[mps]" )
{
    MPS x(4,8);
    x.FillRandu({8,2,8});
    SECTION( "initialization and compatibility" )
    {
        for(int i=0;i<2;i++)
        {
            x.PrintSizes();
            int m=1;
            for(auto t:x.M)
            {
                REQUIRE( t.dim[0]==m );
                REQUIRE( t.dim[1]==2 );
                REQUIRE( t.dim[2]<=x.m );
                m=t.dim[2];
            }            
        }
    }
    SECTION( "Sweep")
    {
        TensorD tx=x;
        while (x.pos<x.length-1)
        {
            x.SweepRight();
            TensorD tx2=x;
            REQUIRE( Norm(tx-tx2)/Norm(tx) < x.tol );
        }
        while (x.pos>=0)
        {
            x.SweepLeft();
            TensorD tx2=x;
            REQUIRE( Norm(tx-tx2)/Norm(tx) < x.tol );
        }
        SECTION( "SetPos")
        {
            x.SetPos(x.length/2);
            TensorD tx2=x;
            REQUIRE( Norm(tx-tx2)/Norm(tx) < x.tol );
        }
        SECTION( "Canonicalize")
        {
            x.Canonicalize();
            TensorD tx2=x;
            REQUIRE( Norm(tx-tx2)/Norm(tx) < x.tol );
        }
    }



    SECTION( "norm" )
    {
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
    }
    SECTION( "MPO operators: * +" )
    {
        SECTION( "random operator" )
        {
            MPO s(x.length,x.m);
            s.FillRandu({x.m,2,2,x.m});
            s.Canonicalize();
            double nr=s.norm();//1<<(x.length/2);
            REQUIRE( s.norm()==Approx(nr) );
            REQUIRE( (s*3).norm()==Approx(3*nr) );
            s+=s;
            REQUIRE( s.norm()==Approx(2*nr) );
        }
        SECTION( "1 operator" )
        {
            MPO s=MPOIdentity(x.length);
            double nr=1<<(x.length/2);
            REQUIRE( s.norm()==Approx(nr) );
            REQUIRE( (s*3).norm()==Approx(3*nr) );
            s+=s;
            REQUIRE( s.norm()==Approx(2*nr) );
        }
    }
}
