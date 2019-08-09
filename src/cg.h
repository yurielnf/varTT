#ifndef CG_H
#define CG_H

#include<iostream>
//*****************************************************************
// Iterative template routine -- CG
//
// CG solves the symmetric positive definite linear
// system Ax=b using the Conjugate Gradient method.
//
// CG follows the algorithm described on p. 15 in the
// SIAM Templates book.
//
// The return value indicates convergence within max_iter (input)
// iterations (0), or no convergence within max_iter iterations (1).
//
// Upon successful return, output arguments have the following values:
//
//        x  --  approximate solution to Ax = b
// max_iter  --  the number of iterations performed before the
//               tolerance was reached
//      tol  --  the residual after the final iteration
//
//*****************************************************************


template < class QOper, class Ket, class Real=double >
int CG(const QOper &A, Ket &x, const Ket &b, int &max_iter, Real &tol)
{
    Real resid;
    Ket p, z, q;
    Real alpha, beta, rho, rho_1;

    Real normb = Norm(b);
    Ket r = b - A*x;

    if (normb == 0.0)
        normb = 1;

    if ((resid = Norm(r) / normb) <= tol) {
        tol = resid;
        max_iter = 0;
        return 0;
    }

    for (int i = 1; i <= max_iter; i++) {
        z = r;//M.solve(r);
        rho = Dot(r, z);

        if (i == 1)
            p = z;
        else {
            beta = rho / rho_1;
            p = z + p*beta;
        }

        q = A*p;
        alpha = rho / Dot(p, q);

        x += p*alpha;
        r -= q*alpha;

        if ((resid = Norm(r) / normb) <= tol) {
            tol = resid;
            max_iter = i;
            return 0;
        }

        rho_1 = rho;
    }

    tol = resid;
    std::cout<<" CG failed, error="<<tol<<" ";
    return 1;
}




#endif // CG_H
