#ifndef CORRECTIONVECTOR_H
#define CORRECTIONVECTOR_H

#include"gmres_m.h"
#include"cg.h"
#include"gmres.h"
//#include"tools.h"

template<class Operator, class Ket>
struct CorrectionVector  // To obtain the Im[c] for the problem: (w + i eta - H) |c> = |a>
{
    const Operator& H;
    Ket a,xI,xR;
    double w,eta,greenI,greenR, tol=1e-4;
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
//        Gmmres<CorrectionVector,Ket> sol(*this,a*(-eta),x0,nInner,nIter,tol);
//        sol.Iterate();
//        cIter=sol.iter;
        cIter=nIter;
        double error=tol;
        xI=x0;
//        CG(*this,xI,a*(-eta),cIter,error);
        GMRES(*this,xI,a,nIter,cIter,error);
        xI*=(-eta);
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
        cIter=sol.iter;
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
        cIter=sol.iter;
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
vector<CorrectionVector<Hamiltonian,Ket>> FindCV(const Hamiltonian& H,const Ket& a,
                                                 const vector<Ket>& cI0,
                                         vector<cmpx> ws,int nIter,double tol=1e-2)
{
    typedef CorrectionVector<Hamiltonian,Ket> CV;
    vector<CV> cvs;
    auto x0=cI0[0];
    for (uint i=0;i<ws.size();i++)
    {
        auto z=ws[i];
        CV cv(H,a);
        cv.tol=tol;
        cv.Solve(z.real(),z.imag(),x0,nIter);
        cvs.push_back(cv);
        x0= (i+1>=cI0.size()) ? cv.xI : cI0.at(i+1);
    }
    return cvs;
}

#include"lanczos.h"
template<class Hamiltonian, class Ket>                                      //Portal method
vector<CorrectionVector<Hamiltonian,Ket>> FindCVL(const Hamiltonian& H,const Ket& a,
                                         vector<cmpx> ws,int nIter,double tol=1e-6)
{
    Lanczos<Hamiltonian,Ket> lan(H,a);
    lan.DoIt(nIter, tol);
    typedef CorrectionVector<Hamiltonian,Ket> CV;
    vector<CV> cvs;
    for ( const auto& x:lan.CorrectionV(ws) )
    {
        CV cv(H,a);
        cv.cIter=lan.iter;
        cv.xR=x[0];
        cv.xI=x[1];
        cvs.push_back(cv);
    }

//    cv.Solve(w,eta,x[1],nIter);
//    cv.cIter+=lan.iter;
    return cvs;
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
