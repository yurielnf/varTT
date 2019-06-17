#ifndef CORRECTIONVECTOR_H
#define CORRECTIONVECTOR_H

#include"gmres_m.h"
//#include"tools.h"

template<class Operator, class Ket>
struct CorrectionVector  // To obtain the Im[c] for the problem: (w + i eta - H) |c> = |a>
{
    const Operator& H;
    Ket a,xI,xR;
    double w,eta,greenI,greenR, tol=1e-4
            ;
    int cIter=0,nInner=512;

    CorrectionVector(const Operator& H,const Ket& a)
        :H(H)
        ,a(a)        
    {}
    Ket operator*(const Ket& x) const
    {
        Ket y1=x*(w*w+eta*eta);
        Ket y21=H*x-x*(2*w);
        Ket y2=H*y21;
        return y1+y2;
    }

    void Solve(double w, double eta,const Ket& x0,int nIter)
    {
        this->w=w;
        this->eta=eta;
        Gmmres<CorrectionVector,Ket> sol(*this,a*(-eta),x0,nInner,nIter,tol);
        sol.Iterate();
        cIter=sol.cIter;
        xI=sol.x;

        xR=(H*xI-xI*w)*(1.0/eta);
        greenR=Dot(a,xR);
        greenI=Dot(a,xI);
//        using cmpx=std::complex<double>;
//        cmpx g=Dot(a,xR+xI*cmpx(0,1));
//        greenR=g.real();
//        greenI=g.imag();
    }
};

//------------------------------------------- Para numeros complejos ----------------------------------

template<class Operator, class Ket>
struct CorrectionVectorC // To obtain the Im[c] for the problem: (w + i eta - H) |c> = |a>
{
    const Operator& H;
    Ket ar,ai,xI,xR;
    double w,eta, tol=1e-4;
    int cIter=0,nInner=512;

    CorrectionVectorC(const Operator& H,const Ket& ar,const Ket& ai)
        :H(H)
        ,ar(ar)
        ,ai(ai)
    {}
    Ket operator*(const Ket& x) const
    {
        Ket y1=x*(w*w+eta*eta);
        Ket y21=H*x-x*(2*w);
        Ket y2=H*y21;
        return y1+y2;
    }

    void Solve(double w, double eta,const Ket& x0,int nIter)
    {
        this->w=w;
        this->eta=eta;
        Gmmres<CorrectionVectorC,Ket> sol(*this,ar*(-eta)+ai*w-H*ai,x0,nInner,nIter,tol);
        sol.Iterate();
        cIter=sol.cIter;
        xI=sol.x;

        xR=(H*xI-xI*w+ai)*(1.0/eta);
    }
};

//------------------------------------------- Otro vector correccion --------------------------------------------
#include<valarray>
using std::valarray;

template<class Operator, class Ket>
struct CorrectionVector2  // To obtain c for the problem: (w + i eta - H) |c> = |a>
{
    const Operator& H;
    Ket a,xI,xR;
    double w,eta,greenI,greenR, tol=1e-4;
    int cIter=0,nInner=256;

    CorrectionVector2(const Operator& H,const Ket& a)
        :H(H)
        ,a(a)
    {}
    valarray<Ket> operator*(const valarray<Ket>& x) const
    {
        Ket y1=w*x[0]-H*x[0] - eta*x[1];
        Ket y2=eta*x[0] + w*x[1]-H*x[1];
        return {y1,y2};
    }

    void Solve(double w, double eta,const Ket& x0R,const Ket& x0I,int nIter)
    {
        this->w=w;
        this->eta=eta;
        valarray<Ket> b={a,a-a},x0={x0R,x0I};
        Gmmres<CorrectionVector2,valarray<Ket> > sol(*this,b,x0,nInner,nIter,tol);
        sol.Iterate();
        cIter=sol.cIter;
        xR=sol.x[0];
        xI=sol.x[1];
        greenR=Dot(a,xR);
        greenI=Dot(a,xI);
    }
};


//-----------------------------------------------------------------------------------

template<class Hamiltonian, class Ket>                                      //Portal method
CorrectionVector<Hamiltonian,Ket> FindCV(const Hamiltonian& H,const Ket& a,const Ket cI0,
                                         double w,double eta,int nIter)
{
    CorrectionVector<Hamiltonian,Ket> cv(H,a);
    cv.Solve(w,eta,cI0,nIter);
    return cv;
}

template<class Hamiltonian, class Ket>                                      //Portal method
CorrectionVectorC<Hamiltonian,Ket> FindCVC(const Hamiltonian& H,const Ket& ar,const Ket& ai,
                                           const Ket cI0,double w,double eta,int nIter)
{
    CorrectionVectorC<Hamiltonian,Ket> cv(H,ar,ai);
    cv.Solve(w,eta,cI0,nIter);
    return cv;
}

template<class Hamiltonian, class Ket>                                      //Portal method
CorrectionVector2<Hamiltonian,Ket> FindCV2(const Hamiltonian& H,const Ket& a,
                                          const Ket c0R,const Ket c0I,double w,double eta,int nIter)
{
    CorrectionVector2<Hamiltonian,Ket> cv(H,a);
    cv.Solve(w,eta,c0R,c0I,nIter);
    return cv;
}

#endif // CORRECTIONVECTOR_H
