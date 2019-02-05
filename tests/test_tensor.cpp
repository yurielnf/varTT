#include "tensor.h"
//#include<iostream>
#include"catch.hpp"

#include<armadillo>

using namespace std;
using namespace arma;

TEST_CASE( "tensor level 1", "[tensor]" )
{
    int d[]={2,3,2};
    TensorD t({d[0],d[1],d[2]});
    t.FillRandu();
    SECTION( "size" )
    {
        REQUIRE( t.v.size() == d[0]*d[1]*d[2] );
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
        t1.Save("t1.txt");
        TensorD t2(t1.dim);
        t2.Load("t1.txt");
        t2.Save("t2.txt");
        REQUIRE( t1==t2 );
    }
    SECTION( "copy/operator=" )
    {
        Tensor<double> t3(t.dim);
        {
            TensorD t2=t;
            REQUIRE( t==t2 );
            t3=t2;
        }
        REQUIRE( t==t3 );
    }
    SECTION( "ReShape" )
    {
        auto t2=t.ReShape(2);
        REQUIRE( t.v==t2.v );
        REQUIRE( t2.dim==Index({6,2}) );
        t2.v[10]=123;
        REQUIRE( t.v[10]==123 );
    }
    SECTION( "operator-/Norm" )
    {
        auto dt=t-t;
        REQUIRE( Norm(dt)<1e-16 );
    }
    SECTION( "operator+" )
    {
        auto t2=-t;
        auto dt2=t2+t;
        REQUIRE( Norm(dt2)<1e-16 );
    }
    SECTION( "matrix transpose" )
    {
        TensorD t2=t.Transpose(1);
        REQUIRE( t2.dim==Index{3,2,2} );
        auto mt=t.ReShape(1);
        auto mt2=t2.ReShape(t2.rank()-1);
        for(int i=0;i<mt.dim[0];i++)
            for(int j=0;j<mt.dim[1];j++)
                REQUIRE( mt[{i,j}]==mt2[{j,i}] );
    }
    SECTION( "matrix multiplication" )
    {
        TensorD t2=t*t;
        REQUIRE( t2.dim==Index{d[0],d[1],d[1],d[2]} );

        mat A(t.v.data(),6,2);     // Checking result against armadillo
        mat B(t.v.data(),2,6);
        mat C=A*B;
        vector<double> data(C.begin(),C.end());

        REQUIRE( data==t2.v );
    }
    SECTION( "matrix decomposition: svd" )
    {
        auto usvt=SVDecomposition(t,2);
        REQUIRE( usvt[0].dim==Index{d[0],d[1],d[2]} );
        REQUIRE( usvt[1].dim==Index{d[2],d[2]} );
        REQUIRE( usvt[2].dim==Index{d[2],d[2]} );

        auto x=usvt[0]*usvt[1]*usvt[2];
        REQUIRE( Norm(x-t)<1e-15 );
    }

}
