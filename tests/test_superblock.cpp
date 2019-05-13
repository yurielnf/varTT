#include"superblock.h"
#include"catch.hpp"

#include<iomanip>
using namespace std;

TEST_CASE( "superblock for mpo", "[superblock]" )
{
    int len=10;
    SECTION( "superblock <x|x>" )
    {
        int m=128;
        MPS x(len,m,MatSVDFixedTol(1e-12));
        x.FillRandu({m,2,m});
        x.Canonicalize();
        x.Normalize();
        REQUIRE( x.norm() == Approx(1) );
        Superblock sb({&x,&x});
        double e=sb.value();
        REQUIRE( e==Approx(1) );
        for(auto i:MPS::SweepPosSec(len))
        {
            sb.SetPos(i);
            REQUIRE( sb.value()==Approx(e) );
        }
    }
    SECTION( "<x|1|x>" )
    {
        int m=8;
        MPS x(len,m);
        x.FillRandu({m,2,m});
        x.Canonicalize();
        x.Normalize();
        REQUIRE( x.norm() == Approx(1) );
        auto op=MPOIdentity(len,2);
        REQUIRE( op.norm()== Approx(sqrt(1<<len)) );
        //        REQUIRE( Norm(op.C)==Approx(1) );
        Superblock sb({&x,&op,&x});
        op=(*sb.mps[1]);
        REQUIRE( op.norm()==Approx(sqrt(1<<len)) );
        REQUIRE( Norm(op.C)==Approx(1) );
        for(int i=0;i<sb.length-1;i++)
        {
            sb.SetPos({i,1});
            REQUIRE( sb.value()==Approx(1) );
        }
    }
    SECTION( "superblock <x|H|x>" )
    {
        int m=128;
        MPS x(len,m,MatSVDFixedDim(m));
        x.FillRandu({m,2,m});
        x.Canonicalize();
        x.Normalize();
        x.PrintSizes("|x>=");
        REQUIRE( x.norm() == Approx(1) );
//        auto op=HamTBExact(len);
        auto op=HamTbAuto(len,false); op.decomposer=MatChopDecompFixedTol(0);
        op.PrintSizes();
        Superblock sb({&x,&op,&x});
        double e=sb.value();
        std::cout<<setprecision(15);
        for(int k=0;k<1000;k++)
        {
            for(auto p:MPS::SweepPosSec(len))
            {
                sb.SetPos(p);
                REQUIRE( sb.value()==Approx(e) );
                REQUIRE( sb.value(1)==Approx(e) );
            }
            if (k%100==0) std::cout<<sb.value()<<" "<<sb.value(1)<<"\n";
        }
    }
}
