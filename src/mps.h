#ifndef MPS_H
#define MPS_H

#include "tensor.h"
#include <iostream>
#include <vector>

class MPS {

public:
  std::vector<TensorD> M;
  TensorD C;
  int length, m;
  double tol = 1e-14;

  MPS(int length, int m) : M(length), length(length), m(m) {
    C == TensorD({1, 1}, {1});
  }

  void FillRandu(Index dim) {
    for (auto &x : M)
      x = TensorD(dim);
    Index dl = dim;
    dl.front() = 1;
    Index dr = dim;
    dr.back() = 1;
    M.front() = TensorD(dl);
    M.back() = TensorD(dr);

    for (TensorD &x : M)
      x.FillRandu();
  }

  void PrintSizes() const {
    for (TensorD t : M) {
      for (int x : t.dim)
        std::cout << " " << x;
      std::cout << ",";
    }
    std::cout << "\n";
  }

  void Canonicalize() {
    C = TensorD({1, 1}, {1});
    pos = -1;
    while (pos < length / 2 - 1)
      SweepRight();
    auto cC = C;
    C = TensorD({1, 1}, {1});
    pos = length - 1;
    while (pos > length / 2 - 1)
      SweepLeft();
    C = cC * C;
    ExtractNorm(C);
  }

  void Normalize() { norm_n = 1; }

  void SweepRight() {
    pos++;
    auto psi = C * M[pos];
    ExtractNorm(psi);
    auto usvt = SVDecomposition(psi, psi.rank() - 1);
    M[pos] = usvt[0];
    C = usvt[1] * usvt[2];
  }

  void SweepLeft() {
    auto psi = M[pos] * C;
    ExtractNorm(psi);
    auto usvt = SVDecomposition(psi, 1);
    M[pos] = usvt[2];
    C = usvt[0] * usvt[1];
    pos--;
  }

private:
  void ExtractNorm(TensorD &psi) {
    double nr = Norm(psi);
    if (nr < tol)
      throw std::logic_error("mps:ExtractNorm() null matrix");
    norm_n *= pow(nr, 1.0 / M.size());
    psi *= 1.0 / nr;

    norm_n *= pow(Norm(C), 1.0 / M.size());
  }

  int pos = -1;
  double norm_n = 1; // norm(MPS)^(1/n)
};

#endif // MPS_H
