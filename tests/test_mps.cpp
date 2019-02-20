#include "mps.h"
#include"catch.hpp"


TEST_CASE( "mps canonization", "[mps]" )
{
    MPS x(2,4);
    x.FillRandu({4,2,4});
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
            x.Canonicalize();
        }
    }
    SECTION( "norm" )
    {
        x.Canonicalize();
        x.Normalize();
        REQUIRE( x.norm() ==Approx(1) );
        REQUIRE( Norm(x.C)==Approx(1) );
    }
    SECTION( "operators * +" )
    {
        x.Canonicalize(); x.Normalize();
        MPS s=MPOIdentity(x.length,2);
        double nr=1<<(x.length/2);
        REQUIRE( s.norm()==Approx(nr) );
        REQUIRE( (s*3).norm()==Approx(3*nr) );
        s+=s;
        REQUIRE( s.norm()==Approx(2*nr) );
    }
}
