#ifndef CADENITAAA_H
#define CADENITAAA_H

struct CadenitaAA
{
    const int nOrb=3;
    const double t=1.11, U=3, U3=10, J=0.838, muCu=0.5;//2.595;

    CadenitaAA(int Lt,bool periodic,double mu)
        :L(Lt/nOrb/2), Lt(Lt),periodic(periodic),mu(mu)
        ,delta({nOrb,nOrb})
        ,hop({nOrb,nOrb})
        ,Umat({2*nOrb,2*nOrb})
    {
        delta.FillZeros();
        hop.FillZeros();
        Umat.FillZeros();
        delta[{1,2}] = delta[{2,1}] = t;
        hop[{0,2}] = -t;
        delta[{2,2}]=-muCu;
        for(int i=0;i<nOrb;i++) delta[{i,i}]-=mu;
        Umat[{toInt(0,0),toInt(0,1)}]=Umat[{toInt(1,0),toInt(1,1)}]=U;
        Umat[{toInt(2,0),toInt(2,1)}]=U3;
        for(int s=0;s<2;s++)
            for(int sp=0;sp<2;sp++)
            {
                double c= (s==sp)? U-3*J : U-2*J;
                Umat[{toInt(0,s),toInt(1,sp)}]=c;
            }
    }

    int toInt(int i, int Ii, int spin) const { return spin+Ii*2+i*nOrb*2; }
    int toInt(int Ii, int spin) const { return spin+Ii*2; }
    MPO Kin() const
    {
        const int m=1;
        MPSSum h(m,MatSVDFixedTol(1e-13));
        for(int i=0;i<L-1+periodic; i++)
            for(int ii=0;ii<nOrb;ii++)
                for(int jj=0;jj<nOrb;jj++)
                {
                    double tt=hop[{ii,jj}];
                    if (tt!=0)
                        for(int s=0;s<2;s++)
                        {
                            int pi=toInt(i,ii,s);
                            int pj=toInt((i+1)%L,jj,s);
                            h += Fermi(pi,Lt,true)*Fermi(pj,Lt,false)*tt ;
                            h += Fermi(pj,Lt,true)*Fermi(pi,Lt,false)*tt ;
                        }
                    double ee=delta[{ii,jj}];
                    if (ee!=0)
                        for(int s=0;s<2;s++)
                        {
                            int pi=toInt(i,ii,s);
                            int pj=toInt(i,jj,s);
                            h += Fermi(pi,Lt,true)*Fermi(pj,Lt,false)*ee ;
                        }
                }
        return h.toMPS();
    }
    MPO Pot() const
    {
        const int m=1;
        MPSSum h(m,MatSVDFixedTol(1e-13));
        for(int i=0;i<L; i++)
            for(int ii=0;ii<nOrb;ii++)
                for(int s=0;s<2;s++)
                    for(int jj=0;jj<nOrb;jj++)
                        for(int sp=0;sp<2;sp++)
                        {
                            int pi=toInt(i,ii,s);
                            int pj=toInt(i,jj,sp);
                            double coeff=Umat[{toInt(ii,s),toInt(jj,sp)}];
                            if (coeff==0) continue;
                            h += Fermi(pi,Lt,true)*Fermi(pi,Lt,false)*           //U n n
                                 Fermi(pj,Lt,true)*Fermi(pj,Lt,false)*coeff;
                        }
        if (J!=0)
        for(int i=0;i<L; i++)
            for(int ii=0;ii<2;ii++)
            {
                int pi=toInt(i,ii,0);
                int pj=toInt(i,1-ii,1);
                int pk=toInt(i,1-ii,0);
                int pl=toInt(i,ii,1);
                h += Fermi(pi,Lt,true)*Fermi(pj,Lt,true)*           // J c+c+ cc
                     Fermi(pk,Lt,false)*Fermi(pl,Lt,false)*J;
                pi=toInt(i,ii,0);
                pj=toInt(i,ii,1);
                pk=toInt(i,1-ii,1);
                pl=toInt(i,1-ii,0);
                h += Fermi(pi,Lt,true)*Fermi(pj,Lt,true)*           // J c+c+ cc
                     Fermi(pk,Lt,false)*Fermi(pl,Lt,false)*J;
            }
        return h.toMPS();
    }
    MPO toMPO() const
    {
        return Kin()+Pot();
    }
private:
    int L,Lt;
    bool periodic;
    double mu;
    TensorD delta, hop, Umat;
};

#endif // CADENITAAA_H
