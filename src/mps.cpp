#include "mps.h"


//---------------------------- Helpers ---------------------------

MPO MPOIdentity(int length,int d)
{
    static std::vector<double> id(d*d);
    MatFillEye(id.data(),d);
    std::vector<TensorD> O(length);
    for(auto& x:O) x=TensorD({1,d,d,1}, id);
    return O;
}
MPO Pauli1Sz(int i,int L)
{
    static TensorD
            one( {1,3,3,1}, {1,0,0,
                             0,1,0,
                             0,0,1} ),
            sz( {1,3,3,1}, {-1,0,0,
                             0,0,0,
                             0,0,1} );
    std::vector<TensorD> O(L);
    for(int j=0;j<L;j++)
        O[j]= (j==i) ? sz
                     : one ;
    return O;
}
MPO Pauli1Sp(int i,int L)
{
    const double r2=sqrt(2);
    static TensorD
            one( {1,3,3,1}, {1,0,0,
                             0,1,0,
                             0,0,1} ),
            sp ( {1,3,3,1}, {0,r2,0,
                             0,0,r2,
                             0,0,0} );
    std::vector<TensorD> O(L);
    for(int j=0;j<L;j++)
        O[j]= (j==i) ? sp
                     : one ;
    return O;
}
MPO Pauli1Sm(int i,int L)
{
    const double r2=sqrt(2);
    static TensorD
            one( {1,3,3,1}, {1,0,0,
                             0,1,0,
                             0,0,1} ),
            sm ( {1,3,3,1}, {0,0,0,
                             r2,0,0,
                             0,r2,0} );
    std::vector<TensorD> O(L);
    for(int j=0;j<L;j++)
        O[j]= (j==i) ? sm
                     : one ;
    return O;
}

MPO SpinSz(int s,int i,int L)    // https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins
{
    int n=2*s+1;
    TensorD one({n,n}), sz({n,n});
    one.FillEye(1);
    sz.FillZeros();
    int c=0;
    for(int m=-s;m<=s;m++,c++)\
        sz[{c,c}]=1.0*m;

    std::vector<TensorD> O(L);
    for(int j=0;j<L;j++)
        O[j]= (j==i) ? TensorD({1,n,n,1},sz.vec())
                     : TensorD({1,n,n,1},one.vec());
    return O;
}
MPO SpinSplus(int s,int i,int L)
{
    int n=2*s+1;
    TensorD one({n,n}), op({n,n});
    one.FillEye(1);
    op.FillZeros();
    int c=0;
    for(int m=-s;m<s;m++,c++)\
        op[{c+1,c}]=sqrt(s*(s+1)-m*(m+1));

    std::vector<TensorD> O(L);
    for(int j=0;j<L;j++)
        O[j]= (j==i) ? TensorD({1,n,n,1},op.vec())
                     : TensorD({1,n,n,1},one.vec());
    return O;
}
MPO SpinSminus(int s,int i,int L)
{
    int n=2*s+1;
    TensorD one({n,n}), op({n,n});
    one.FillEye(1);
    op.FillZeros();
    int c=0;
    for(int m=-s;m<s;m++,c++)\
        op[{c,c+1}]=sqrt(s*(s+1)-m*(m+1));

    std::vector<TensorD> O(L);
    for(int j=0;j<L;j++)
        O[j]= (j==i) ? TensorD({1,n,n,1},op.vec())
                     : TensorD({1,n,n,1},one.vec());
    return O;
}


MPO SpinSzF(int s,int i,int L)    // for spin 3/2, s=3
{
    int n=s+1;
    TensorD one({n,n}), sz({n,n});
    one.FillEye(1);
    sz.FillZeros();
    int c=0;
    for(int m=-s;m<=s;m+=2,c++)\
        sz[{c,c}]=0.5*m;

    std::vector<TensorD> O(L);
    for(int j=0;j<L;j++)
        O[j]= (j==i) ? TensorD({1,n,n,1},sz.vec())
                     : TensorD({1,n,n,1},one.vec());
    return O;
}
MPO SpinSplusF(int s,int i,int L)
{
    int n=s+1;
    TensorD one({n,n}), op({n,n});
    one.FillEye(1);
    op.FillZeros();
    int c=0;
    for(int m=-s;m<s;m+=2,c++)\
        op[{c+1,c}]=sqrt(0.25*s*(s+2)-0.25*m*(m+2));

    std::vector<TensorD> O(L);
    for(int j=0;j<L;j++)
        O[j]= (j==i) ? TensorD({1,n,n,1},op.vec())
                     : TensorD({1,n,n,1},one.vec());
    return O;
}
MPO SpinSminusF(int s,int i,int L)
{
    int n=s+1;
    TensorD one({n,n}), op({n,n});
    one.FillEye(1);
    op.FillZeros();
    int c=0;
    for(int m=-s;m<s;m+=2,c++)\
        op[{c,c+1}]=sqrt(0.25*s*(s+2)-0.25*m*(m+2));

    std::vector<TensorD> O(L);
    for(int j=0;j<L;j++)
        O[j]= (j==i) ? TensorD({1,n,n,1},op.vec())
                     : TensorD({1,n,n,1},one.vec());
    return O;
}



MPO Fermi(int i, int L, bool dagged)
{
    static TensorD
            one( {1,2,2,1}, {1,0,0,1} ),
            sg( {1,2,2,1}, {1,0,0,-1} ),
            cd( {1,2,2,1}, {0,1,0,0} ),
            c ( {1,2,2,1}, {0,0,1,0} );

    auto fe=dagged ? cd : c;
    std::vector<TensorD> O(L);
    for(int j=0;j<L;j++)
    {
        O[j]= (j <i) ? sg :
                       (j==i) ? fe :
                                one ;
    }
    return O;
}

MPO ElectronHoleMPO(int L)
{
    static stdvec j={0,1,1,0}, jm={0,-1,-1,0};
    std::vector<TensorD> O(L);
    for(int i=0;i<L;i++)
    {
        if (i%2==0)
            O[i]=TensorD({1,2,2,1},j);
        else
            O[i]=TensorD({1,2,2,1},jm);
    }
    return O;
}

MPO HamS1(int L,bool periodic)
{
    const int m=10;
    MPSSum h(m,MatSVDFixedTol(1e-13));
    for(int i=0;i<L-1+periodic; i++)
    {
        h += Pauli1Sz(i,L) * Pauli1Sz((i+1)%L,L) ;
        h += Pauli1Sp(i,L) * Pauli1Sm((i+1)%L,L) * 0.5;
        h += Pauli1Sm(i,L) * Pauli1Sp((i+1)%L,L) * 0.5;
    }
    return h.toMPS();
}
MPO HamS(int s,int L,bool periodic)
{
    const int m=3;
    MPSSum h(m,MatSVDFixedTol(1e-13));
    for(int i=0;i<L-1+periodic; i++)
    {
        h += SpinSz(s,i,L) * SpinSz(s,(i+1)%L,L) ;
        h += SpinSplus(s,i,L) * SpinSminus(s,(i+1)%L,L) * 0.5;
        h += SpinSminus(s,i,L) * SpinSplus(s,(i+1)%L,L) * 0.5;
    }
    return h.toMPS();
}

MPO HamSFermi(int s,int L,bool periodic)
{
    const int m=3;
    MPSSum h(m,MatSVDFixedTol(1e-13));
    for(int i=0;i<L-1+periodic; i++)
    {
        h += SpinSzF(s,i,L) * SpinSzF(s,(i+1)%L,L) ;
        h += SpinSplusF(s,i,L) * SpinSminusF(s,(i+1)%L,L) * 0.5;
        h += SpinSminusF(s,i,L) * SpinSplusF(s,(i+1)%L,L) * 0.5;
    }
    return h.toMPS();
}

MPO SpinFlipGlobal(int length)
{
    const stdvec sf={0,0,1,
                     0,1,0,
                     1,0,0} ;
    std::vector<TensorD> O(length);
    for(uint i=0;i<O.size();i++)
            O[i]=TensorD({1,3,3,1}, sf);
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
