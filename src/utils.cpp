#include "utils.h"
#include <armadillo>

using namespace std;
using namespace arma;

void MatTranspose(const double *X, double *result, int nrow, int ncol) {
  const mat mX((double *const)X, nrow, ncol, false);
  mat res(result, ncol, nrow, false);
  res = mX.t();
}

void MatMul(const double *mat1, const double *mat2, double *result, int nrow1,
            int ncol1, int ncol2) {
  const mat m1((double *const)mat1, nrow1, ncol1, false);
  const mat m2((double *const)mat2, ncol1, ncol2, false);
  mat res(result, nrow1, ncol2, false);
  res = m1 * m2;
}

void MatFullDiag(double *const X, int n, double *evec, double *eval) {
  const mat mX(X, n, n, false);
  mat mevec(evec, n, n, false);
  vec meval(eval, n, false);
  eig_sym(meval, mevec, mX);
}

void MatSVD(const double *X, int m, int n, double *U, double *S, double *V) {
  const mat mX((double *const)X, m, n, false);
  int ns = std::min(m, n);
  mat mU(U, m, ns, false);
  vec vS(S, ns, false);
  mat mV(V, n, ns, false);
  svd_econ(mU, vS, mV, mX);
}
