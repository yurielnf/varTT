#include"superblock.h"
#include"catch.hpp"

TEST_CASE( "superblock for mpo", "[superblock]" )
{
    int len=2, m=3;
    MPS x(len,m);
    x.FillRandu({m,2,m});
    SECTION( "<x|1|x>" )
    {
        x.Canonicalize();
        x.Normalize();
        REQUIRE( x.norm() == Approx(1) );
        auto op=MPOIdentity(len,2);
        op.Canonicalize();
        REQUIRE( op.norm()== Approx(sqrt(1<<len)) );
//        REQUIRE( Norm(op.C)==Approx(1) );
        Superblock sb({x,op,x});
        sb.Canonicalize();
        op=sb.mps[1];
        REQUIRE( op.norm()== Approx(sqrt(1<<len)) );
//        REQUIRE( Norm(op.C)==Approx(1) );
        REQUIRE( sb.value()==Approx(1) );
    }
}
