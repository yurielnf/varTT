#include "utils.h"
#include<armadillo>

using namespace std;
using namespace arma;

double VecNorm(const double*  x, int nelem)
{
    const vec mx((double* const)x,nelem,false);
    return arma::norm(mx);
}

void VecPlusInplace(double* x,const double* y,int n)
{
    vec mx(x,n,false,true);
    const vec my((double* const)y,n,false);
    mx+=my;
}

void Vec_xa_Inplace(double* x,const double* y,double a,int n)
{
    vec mx(x,n,false,true);
    const vec my((double* const)y,n,false);
    mx+=my*a;
}

void MatTranspose(const double*  X, double *result, int nrow, int ncol)
{
    const mat mX((double* const)X,nrow,ncol,false);
    mat res(result,ncol,nrow,false,true);
    res=mX.t();
}

void MatMul(const double*  mat1, const double*  mat2, double *result, int nrow1, int ncol1, int ncol2)
{
    const mat m1((double* const)mat1,nrow1,ncol1,false);
    const mat m2((double* const)mat2,ncol1,ncol2,false);
    mat res(result,nrow1,ncol2,false);
    res=m1*m2;
}

void MatMulT(const double*  mat1, const double*  mat2, double *result, int nrow1, int ncol1, int nrow2)
{
    const mat m1((double* const)mat1,nrow1,ncol1,false);
    const mat m2((double* const)mat2,nrow2,ncol1,false);
    mat res(result,nrow1,nrow2,false,true);
    res=m1*m2.t();
}

void MatTMul(const double*  mat1, const double*  mat2, double *result, int nrow1, int ncol1, int ncol2)
{
    const mat m1((double* const)mat1,nrow1,ncol1,false);
    const mat m2((double* const)mat2,nrow1,ncol2,false);
    mat res(result,ncol1,ncol2,false,true);
    res=m1.t()*m2;
}

void MatFullDiag(double* const X,int n,double *evec,double *eval)
{
    const mat mX(X,n,n,false);
    mat mevec(evec,n,n,false,true);
    vec meval(eval,n,false);
    eig_sym(meval,mevec,mX);
}

void MatFullDiagGen(double * const X, double * const O, int n, double *evec, double *eval)
{
    const mat mX(X,n,n,false);
    const mat mO(O,n,n,false);
    mat y,mevec(evec,n,n,false,true);
    vec meval(eval,n,false);
    mat Ri;
    try {Ri=chol(mO).i(); }
    catch (std::exception e){
        mO.print("Chol failed, O=");}
    eig_sym(meval,y,Ri.t()*mX*Ri);
    mevec=Ri*y;
}


array<stdvec,2> MatSVD(bool is_right,const double*  X, int n1,int n2,double tol,int Dmax)
{
    const mat mX((double* const)X,n1,n2,false);
    mat U,V;
    vec s;
    svd_econ(U, s, V, mX);
    double sum0=0;
    for(int i=s.size()-1;i>=0;i--) sum0+=s[i]*s[i];
    int D=std::min(Dmax,(int)s.size());
    if (D==0)
    {
        D=s.size();
        for(uint i=0;i<D;i++)
            if(i>0 && fabs(s[i])<=tol) {D=i;break;}
    }
    double sum=0;
    for(int i=D-1;i>=0;i--) sum+=s[i]*s[i];
    if (sum<=0.0) sum=1;
    mat Ua=U.head_cols(D);
    mat Sa=diagmat(s.head(D)/sqrt(sum/sum0));
    mat Vt=V.head_cols(D).t();
    if (is_right)
        return {conv_to<stdvec>::from(vectorise(Ua*Sa)),
                conv_to<stdvec>::from(vectorise(Vt))    };
    else
        return {conv_to<stdvec>::from(vectorise(Ua)),
                conv_to<stdvec>::from(vectorise(Sa*Vt)) };
}

static bool abs_compare(double a, double b)
{
    return (std::abs(a) < std::abs(b));
}

vector<int> FindNonZeroCols(const double*  X, int n1,int n2,double tol)
{
    vector<int> cols;
    for(int j=0;j<n2;j++)
    {
        double mc=*std::max_element(X+j*n1,X+(j+1)*n1,abs_compare);
        if ( std::abs(mc) > tol ) cols.push_back(j);
    }
    if (cols.empty())
        //throw runtime_error("FindNonZeroCols empty");
        cols.push_back(0);
    return cols;
}

mat MatSelectCols(const mat& A, const vector<int>& cols)
{
    mat B(A.n_rows,cols.size());
    for(uint j=0;j<cols.size();j++)
        B.col(j)=A.col(cols[j]);
    return B;
}

array<mat,2> MatChopDecompArmaByCol(const mat& mX,double tol)
{
    int n1=mX.n_rows, n2=mX.n_cols;
    mat U(n2,n2,fill::eye);
    auto cols=FindNonZeroCols(mX.memptr(),n1,n2,tol);
    mat Xa=MatSelectCols(mX,cols);
    mat Ua=MatSelectCols(U ,cols).t();
    return {Xa,Ua};
}

array<stdvec,2> MatChopDecomp(bool is_right, const double*  X, int n1, int n2, double tol)
{
    const mat mX((double* const)X,n1,n2,false);
//    tol=max( tol, mX.max()*numeric_limits<double>::epsilon());
    if (is_right)
    {
//        stdvec vx(X,X+n1*n2);
//        stdvec vu(n1*n1);
//        MatFillEye(vu.data(),n1);
//        return {vu,vx};

        auto ab=MatChopDecompArmaByCol(mX.t(),tol);
        mat a=ab[1].t();
        mat b=ab[0].t();
        return {stdvec(a.begin(),a.end()),
                stdvec(b.begin(),b.end())};
    }
    else
    {
//        stdvec vx(X,X+n1*n2);
//        stdvec vu(n2*n2);
//        MatFillEye(vu.data(),n2);
//        return {vx,vu};

        auto ab=MatChopDecompArmaByCol(mX,tol);
        return {stdvec(ab[0].begin(),ab[0].end()),
                stdvec(ab[1].begin(),ab[1].end())};
    }
}

array<stdvec,2> MatQRDecomp(bool is_right, const double*  X, int n1, int n2)
{
    const mat mX((double* const)X,n1,n2,false);
    if (is_right)
    {
        mat Q,R;
        qr_econ(Q,R,mX.t());
        mat a=R.t();
        mat b=Q.t();
        return {stdvec(a.begin(),a.end()),
                stdvec(b.begin(),b.end())};
    }
    else
    {
        mat Q,R;
        qr_econ(Q,R,mX);
        return {stdvec(Q.begin(),Q.end()),
                stdvec(R.begin(),R.end())};
    }
}



MatDensityFixedDimDecomp::MatDensityFixedDimDecomp(stdvec rho,int _m)
    :m(_m)
{
    vec eval;
    mat evec;
    int n=sqrt(rho.size());
    mat rhom(rho.data(),n,n,false);
    eig_sym(eval,evec,rhom);
    m=std::min(n,m);
    double sum=0;
    for(uint i=0;i<m;i++) sum+=eval[n-m+i];
    if (fabs(1.0-sum)>1e-9) std::cout<<" peso descartado "<<1.0-sum<<" tr(rho)="<<arma::trace(rhom)<<"\n";
    mat evecT=evec.tail_cols(m);
    rot=stdvec(evecT.begin(),evecT.end());
}

std::array<stdvec,2> MatDensityFixedDimDecomp::operator()(bool is_right, const double*  X, int n1,int n2) const
{
    const mat mX((double* const)X,n1,n2,false);
    int mr=rot.size()/m;
    const mat mrot((double* const) rot.data(),mr,m,false);
    if (!is_right)
    {
        if (mr!=n1)
            throw std::invalid_argument("MatDensityDecomp left rotation incompatible");
        mat mrott=mrot.t();
        mat C=mrott*mX;
        return {stdvec(mrot.begin(),mrot.end()),
                stdvec(C.begin(),C.end())};
    }
    else
    {
        if (mr!=n2)
            throw std::invalid_argument("MatDensityDecomp right rotation incompatible");

        mat mrott=mrot.t();
        mat C=mX*mrot;
        return {stdvec(C.begin(),C.end()),
                stdvec(mrott.begin(),mrott.end())};
    }
}

double Entropy(const double* eval,int n_elem)
{
    double sum=0;
    for(uint i=0;i<n_elem;i++)
        if (fabs(eval[i])>1e-12)
            sum-=eval[i]*log2(eval[i]);
    return sum;
}
double EntropyRenyi(const double* eval,int n_elem, double q)
{
    if (q==1) return Entropy(eval,n_elem);
    double sum=0;
    for(uint i=0;i<n_elem;i++)
        if (fabs(eval[i])>1e-12)
            sum+=pow(eval[i],q);
    return log2(sum)*1.0/(1-q);
}

