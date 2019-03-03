#include"superblock.h"
#include"catch.hpp"


int Square(int x)
{
    return x*x;
}

TEST_CASE( "dmrg tight-binding", "[dmrg_tb]" )
{
    int len=10, m=8;
    MPS x(len,m);
    x.FillRandu({m,2,m});
    SECTION( "<x|1|x>" )
    {
        x.Canonicalize();
        x.Normalize();
        REQUIRE( x.norm() == Approx(1) );
        auto op=MPOIdentity(len);
        REQUIRE( op.norm()== Approx(sqrt(1<<len)) );
//        REQUIRE( Norm(op.C)==Approx(1) );
        Superblock sb({x,op,x});
        op=sb.mps[1];
        REQUIRE( op.norm()==Approx(sqrt(1<<len)) );
        REQUIRE( Norm(op.C)==Approx(1) );
        for(int i=0;i<sb.length-1;i++)
        {
            sb.SetPos(i);
            REQUIRE( sb.value()==Approx(1) );
        }
    }
}
