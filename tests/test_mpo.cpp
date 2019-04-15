#include"catch.hpp"
#include"mps.h"

TEST_CASE("auto mpo","[mpo]")
{
    int len=10;
    SECTION( "MPO operators: * +" )
    {
        SECTION( "random operator" )
        {
            int m=128;
            MPO s(len,128);
            s.FillRandu({m,2,2,m});
            s.Canonicalize();
            double nr=s.norm();//1<<(x.length/2);
            REQUIRE( s.norm()==Approx(nr) );
            REQUIRE( (s*3).norm()==Approx(3*nr) );
            s+=s;
            REQUIRE( s.norm()==Approx(2*nr) );
        }
        SECTION( "1 operator" )
        {
            MPO s=MPOIdentity(len);
            double nr=pow(2,len/2.);
            REQUIRE( s.norm()==Approx(nr) );
            REQUIRE( (s*3).norm()==Approx(3*nr) );
            s+=s;
            REQUIRE( s.norm()==Approx(2*nr) );
        }
    }
    SECTION( "MPSSum" )
    {
        MPO s=MPOIdentity(len);
        double nr=pow(2,len/2.);
        REQUIRE( s.norm()==Approx(nr) );
        MPSSum sum(50,MatSVDFixedTol(1e-14));
        for(int i=0;i<50;i++)
            sum+=s;
        REQUIRE( sum.toMPS().norm()==Approx(50*nr) );
        REQUIRE( sum.toMPS().MaxVirtDim()==s.MaxVirtDim() );
    }
    SECTION( "MPO for TB Hamiltonian" )
    {
        int len=10;
        MPO he=HamTBExact(len);
        he.PrintSizes("HtbExact=");
        MPO h=HamTbAuto(len,false);
        h.PrintSizes("Htb=");
        TensorD the=he;
        TensorD th=h;
        REQUIRE( Norm(the-th)/Norm(the) < 1e-13 );
    }
}
