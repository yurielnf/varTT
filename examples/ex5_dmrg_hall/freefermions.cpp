#include "freefermions.h"
#include<armadillo>
#include<fstream>

using namespace arma;
using namespace std;

//------------------------------------------------- entropia no interactuante, en el toro ------------------------

mat Ham1D(size_t L,bool periodic)
{
    mat h(L,L,fill::zeros);
    for(size_t i=0;i<L-1+periodic;i++)
        h(i,(i+1)%L)=1;
    return h+h.t();
}

mat Ham2D(size_t Lx,size_t Ly,bool periodic)
{
    size_t dxs[]={1,0};
    size_t dys[]={0,1};
    mat h(Lx*Ly,Lx*Ly,fill::zeros);
    for(size_t ix=0;ix<Lx-1+periodic;ix++)
        for(size_t iy=0;iy<Ly;iy++)
            for(int d=0;d<2;d++)
                {
                    size_t dx=dxs[d];
                    size_t dy=dys[d];
                    size_t i=ix*Ly+iy;
                    size_t jx=(ix+dx+Lx)%Lx;
                    size_t jy=(iy+dy+Ly)%Ly;
                    size_t j=jx*Ly+jy;
                    h(i,j)=h(j,i)=1;
                }
    return h;
}

double Entropia(vec eval)
{
    double sum=0;
    for(size_t i=0;i<eval.size();i++)
        if (fabs(eval(i))>1e-10  && fabs(1-eval(i))>1e-10)
            sum-=eval[i]*log2(eval[i])+(1-eval[i])*log2(1-eval[i]);
    return sum;
}

double EntropiaR(vec eval,double q)
{
    if (q==1) return Entropia(eval);
    double sum=0;
    for(size_t i=0;i<eval.size();i++)
        if (fabs(eval(i))>1e-10  && fabs(1-eval(i))>1e-10)
            sum+= q==1.0 ? -eval[i]*log2(eval[i])-(1-eval[i])*log2(1-eval[i])
                         : log2( pow(eval[i],q)+pow(1-eval[i],q) );
    return q==1.0? sum : sum/(1.0-q);
}

void TestRenyiLibre1D(const Parameters &param)
{
    int Lx=param.freeFermionLx, nPart=Lx/(4*M_PI);
    mat evec;
    vec eval;
    mat h=Ham1D(Lx,false);
    eig_sym(eval,evec, h );
    for(int i=-3;i<4;i++)
       cout<<i<<" "<<eval[nPart-1+i]<<"\n";
    cout<<endl<<endl;
    mat u=evec.cols(0,nPart-1);
    mat c=u*u.t();
    ofstream out(string("renyi1dlibre")+to_string(Lx)+".dat");
    out<<"r ";
    for(auto q:param.renyi_q) out<<"q="<<q<<" ";
    out<<endl;
    for(int r=0;r<=Lx/2;r++)
    {
        mat cr=c.submat(0,0,r,r);
        vec eval=eig_sym(cr);
        out<<r+1;
        for(double q:param.renyi_q)
            out<<" "<<EntropiaR(eval,q);
        out<<endl;
    }
}

void TestRenyiLibre2D(const Parameters &param)
{
    int factor=1, L=100*factor;
    int Ly=factor*10,
            Lx=factor*2*M_PI*L/Ly,
            nPart=1/(4*M_PI)*Lx*Ly;
//    int Lx=50*factor, Ly=40*factor, nPart=Lx*Ly/(4*M_PI)-3;
    cout<<Lx<<" "<<Ly<<" "<<nPart<<endl;
    mat evec;
    vec eval;
    mat h=Ham2D(Lx,Ly,false);
    eig_sym(eval,evec, h );
    for(int i=-3;i<4;i++)
       cout<<i<<" "<<eval[nPart-1+i]<<endl;
    cout<<endl<<endl;
    mat u=evec.cols(0,nPart-1);
    mat c=u*u.t();
    ofstream out(string("renyi2dlibre")+to_string(Lx)+"_"+to_string(Ly)+".dat");
    out<<"r ";
    for(auto q:param.renyi_q) out<<"q="<<q<<" ";
    out<<endl;
    for(int r=Ly-1;r<=Lx*Ly/2;r+=Ly)
    {
        mat cr=c.submat(0,0,r,r);
        vec eval=eig_sym(cr);
        out<<(r+1)/Ly;
        for(double q:param.renyi_q)
            out<<" "<<EntropiaR(eval,q);
        out<<endl;
    }
}
