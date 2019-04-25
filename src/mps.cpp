#include "mps.h"


//---------------------------- Helpers ---------------------------

MPO MPOIdentity(int length)
{
    std::vector<TensorD> O(length);
    std::vector<double> id={1,0,0,1};
    for(auto& x:O) x=TensorD({1,2,2,1}, id);
    return O;
}
MPO Fermi(int i, int L, bool dagged)
{
    static TensorD
            id( {1,2,2,1}, {1,0,0,1} ),
            sg( {1,2,2,1}, {1,0,0,-1} ),
            cd( {1,2,2,1}, {0,1,0,0} ),
            c ( {1,2,2,1}, {0,0,1,0} );

    auto fe=dagged ? cd : c;
    std::vector<TensorD> O(L);
    for(int j=0;j<L;j++)
    {
        O[j]= (j <i) ? sg :
                       (j==i) ? fe :
                                id ;
    }
    return O;
}
MPO MPOEH(int length)
{
    std::vector<TensorD> O(length);
    std::vector<double> eh={1.3,-2,-1,0}, id={1,0,0,1};
    for(uint i=0;i<O.size();i++)
        if(i%2==0)
            O[i]=TensorD({1,2,2,1}, eh);
        else
            O[i]=TensorD({1,2,2,1}, id);
    return O;
}

MPO HamTbAuto(int L,bool periodic)
{
    const int m=4;
    MPSSum h(m,MatSVDFixedTol(1e-13));
    for(int i=0;i<L-1+periodic; i++)
    {
        h += Fermi(i,L,true)*Fermi((i+1)%L,L,false)*(-1.0) ;
        h += Fermi((i+1)%L,L,true)*Fermi(i,L,false)*(-1.0) ;
    }
    return h.toMPS();
}

MPO HamHubbardAuto(int L)
{
    const int m=4;
    MPSSum h(m,MatSVDFixedTol(1e-13));
    for(int i=0;i<L-1; i++)
    {
        h += Fermi(i,L,true)*Fermi(i+1,L,false)*(-1.0) ;
        h += Fermi(i+1,L,true)*Fermi(i,L,false)*(-1.0) ;
    }
    return h.toMPS();
}

MPO HamTBExact(int L)
{
    static TensorD
            I ={{2,2},{1,0,0,1}},
            c ={{2,2},{0,0,1,0}},
            sg={{2,2},{1,0,0,-1}},
            o ={{2,2},{0,0,0,0}};
    TensorD cd=c.t(),n=cd*c, H=o;
    auto fv1=flat({I, H , c*sg*(-1.0), cd*sg, o});
    auto fv2=flat(   { I, H , c*sg*(-1.0), cd*sg, o,      //H
                       o, I , o    , o          , o,      //I
                       o, cd, o    , o          , o,      //cd
                       o, c , o    , o          , o,      //c
                       o, n , o    , o          , I,      //nT
                     }  );
    auto fv3=flat({H,I,cd,c,n});

    std::vector<TensorD> O(L);
    O[0]={ {2,2,5,1}, fv1 };
    for(int i=1;i<L-1;i++)
        O[i]={ {2,2,5,5}, fv2 };
    O[L-1]=TensorD( {2,2,1,5},fv3 );
    for(TensorD& x:O) x=x.Reorder("ijkl","lijk");
    return MPS(O,5,MatChopDecompFixedTol(0))*(-1.0);
}
