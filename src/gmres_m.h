#ifndef GMRES_M
#define GMRES_M

#include<vector>
#include<cmath>
#include<iostream>

using std::vector;

template<class Operator, class Ket>
struct Arnoldi
{
    const Operator& A;
    Operator* M;
    vector<Ket> v;    // Orthonormal basis for the Krylov space
    vector<vector<double>> h; // the Hessenberg matrix by columns
    int nIter=0;

    Arnoldi(const Operator& A, const Ket& r0,Operator* M=nullptr)
        :A(A), M(M)
    {
        v.push_back( r0*(1.0/Norm(r0)) );
    }
    void Iterate()
    {
        Ket w=A*v[nIter]; nIter++;
        //if (M!=nullptr) w=arma::spsolve(*M,w);
        vector<double> h(nIter+1);
        for(int k=0;k<nIter;k++)
        {
            h[k]=Dot(w,v[k]);
            w-=v[k]*h[k];
        }
        h[nIter]=Norm(w);
        v.push_back( w*(1.0/h[nIter]) );
        this->h.push_back(h);
    }
};

struct GivensRot
{
    double co,si;
    GivensRot(double x,double y)
    {
        if(y == double(0.0)) {
            co = 1.0;
            si = 0.0;
        } else if (abs(y) > abs(x)) {
            double tmp = x / y;
            si = double(1.0) / sqrt(double(1.0) + tmp*tmp);
            co = tmp*si;
        } else {
            double tmp = y / x;
            co = double(1.0) / sqrt(double(1.0) + tmp*tmp);
            si = tmp*co;
        }
//        double delta=sqrt(x*x+y*y);
//        co=x/delta;
//        si=y/delta;
    }
    inline void Apply(double v[],int pos) const
    {
        double h1  = co*v[pos] + si*v[pos+1];
        double h2  =-si*v[pos] + co*v[pos+1];
        v[pos]  = h1;
        v[pos+1]= h2;
    }
};

inline vector<double> SolveUpperBackware(const vector<vector<double>>& U,
                                         const vector<double> b)
{
    int n=U.size();
    vector<double> y(n);
    for (int k = n-1; k >=0; k--) {
        double sum=0;
        for(int i=k+1;i<n;++i) sum+=U[i][k]*y[i];  //U is a set of columns
        y[k]=(b[k]-sum)/U[k][k];
    }
    return y;
}

template<class Operator, class Ket>
struct Gmmres
{
    const Operator& A;
    Operator *M;
    Ket b,x;
    int m,nMaxIter,iter=0;
    double tol,error;

    Gmmres(const Operator& A, const Ket&b, const Ket& x0,
           int nInnerIter,int nMaxIter,double tol,Operator* M=nullptr)
    :A(A)
    ,M(M)
    ,b(b)
    ,x(x0)
    ,m(nInnerIter)
    ,nMaxIter(nMaxIter)
    ,tol(tol)
    {}    

    void Iterate()
    {
        Ket res=b-A*x;
        while (!IterateInner(res) && iter<nMaxIter)
            res=b-A*x;

//        if (error<tol) std::cout<<"cIter="<<cIter<<" error="<<error<<"; ";
        if (error>tol) std::cout<<"Gmres failed! error="<<error<<"; ";
    }
private:
    bool IterateInner(const Ket& res)
    {
        vector<double> s={ Norm(res) };
        Arnoldi<Operator,Ket> arn(A,res,M);
        vector<GivensRot> J;
        vector<vector<double>> H;
        for(int i=0;i<m;i++)
        {
            arn.Iterate(); iter++; s.push_back(0);
            auto hi=arn.h[i];
            for (int k = 0; k < i; ++k)
                J[k].Apply(hi.data(),k);

            GivensRot rot={hi[i], hi[i+1]};
            rot.Apply(hi.data(),i);
            rot.Apply(s.data(),i);
            J.push_back(rot); H.push_back(hi);
            error=fabs(s[i+1]);
            //std::cout<<cIter<<" error="<<error<<std::endl;
            if(error<tol || iter>=nMaxIter) break;
        }
        auto y=SolveUpperBackware(H,s);
        for(uint i=0;i<J.size();++i) x+=arn.v[i]*y[i];
        return error<tol;
    }
};

//---------------------------------- portal -------------------------

template<class Hamiltonian, class Ket>                                      //Portal method
Gmmres<Hamiltonian,Ket> SolveGMMRES(const Hamiltonian& H,const Ket& b,const Ket x0,
                                         int nIter, double tol)
{
    Gmmres<Hamiltonian,Ket> sol(H,b,x0,nIter,nIter,tol);
    sol.Iterate();
    return sol;
}

#endif // GMRES_M

