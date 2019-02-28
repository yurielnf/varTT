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
        MPS x(6,8);
        x.FillRandu({8,2,8});
        TensorD tx=x;
        while (x.pos<x.length-2)
        {
            x.SweepRight();
            TensorD tx2=x;
            REQUIRE( Norm(tx-tx2)/Norm(tx) < 1e-14 );
        }
        while (x.pos>0)
        {
            x.SweepLeft();
            TensorD tx2=x;
            REQUIRE( Norm(tx-tx2)/Norm(tx) < 1e-14 );
        }
        SECTION( "SetPos")
        {
            x.SetPos(x.length/2);
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
            double nr=pow(2,x.length/2.);
            REQUIRE( s.norm()==Approx(nr) );
            REQUIRE( (s*3).norm()==Approx(3*nr) );
            s+=s;
            REQUIRE( s.norm()==Approx(2*nr) );
        }
    }
    SECTION( "MPSSum" )
    {
        MPO s=MPOIdentity(x.length);
        double nr=pow(2,x.length/2.);
        REQUIRE( s.norm()==Approx(nr) );
        MPSSum sum(100);
        for(int i=0;i<100;i++)
            sum+=s;
        REQUIRE( sum.toMPS().norm()==Approx(100*nr) );
        sum.toMPS().PrintSizes();
    }
}
