#ifndef GMRES_H
#define GMRES_H

#include<vector>
#include<iostream>
using stdvec=std::vector<double>;

//*****************************************************************
// Iterative template routine -- GMRES
//
// GMRES solves the unsymmetric linear system Ax = b using the
// Generalized Minimum Residual method
//
// GMRES follows the algorithm described on p. 20 of the
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

#include <math.h>


template<class Real>
void GeneratePlaneRotation(Real &dx, Real &dy, Real &cs, Real &sn)
{
    if (dy == 0.0) {
        cs = 1.0;
        sn = 0.0;
    } else if (abs(dy) > abs(dx)) {
        Real temp = dx / dy;
        sn = 1.0 / sqrt( 1.0 + temp*temp );
        cs = temp * sn;
    } else {
        Real temp = dy / dx;
        cs = 1.0 / sqrt( 1.0 + temp*temp );
        sn = temp * cs;
    }
}


template<class Real>
void ApplyPlaneRotation(Real &dx, Real &dy, Real &cs, Real &sn)
{
    Real temp  =  cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = temp;
}

template < class Ket >
void UpdateSolution(Ket &x, int k, stdvec &h, stdvec &s, std::vector<Ket> v)
{
    int m1=s.size();
    stdvec y(s);

    // Backsolve:
    for (int i = k; i >= 0; i--) {
        y.at(i) /= h.at(i+m1*i);
        for (int j = i - 1; j >= 0; j--)
            y.at(j) -= h.at(j+m1*i) * y.at(i);
    }

    for (int j = 0; j <= k; j++)
        x += v[j] * y.at(j);
}


template < class Operator, class Ket, class Real >
int
GMRES(const Operator &A, Ket &x, const Ket &b, const int &m, int &max_iter, Real &tol)
{
    Real resid;
    int i, j = 1, k, m1=m+1;
    stdvec s(m1), cs(m1), sn(m1);
    stdvec H( (m1)*(m1) );
    Ket w;

    Real normb = Norm(b);
    Ket r = b - A * x;
    Real beta = Norm(r);

    if (normb == 0.0)
        normb = 1;

    if ((resid = Norm(r) / normb) <= tol) {
        tol = resid;
        max_iter = 0;
        return 0;
    }

    std::vector<Ket> v(m+1);

    while (j <= max_iter) {
        v[0] = r * (1.0 / beta);    // ??? r / beta
        for(auto& x:s) x=0;
        s.at(0) = beta;

        for (i = 0; i < m && j <= max_iter; i++, j++) {
            w = A * v[i];
            for (k = 0; k <= i; k++) {
                H.at(k+i*m1) = Dot(w, v[k]);
                w -=  v[k]*H.at(k+i*m1);
            }
            H.at(i+1+i*m1) = Norm(w);
            v[i+1] = w * (1.0 / H.at(i+1+i*m1)); // ??? w / H(i+1, i)

            for (k = 0; k < i; k++)
                ApplyPlaneRotation(H.at(k+i*m1), H.at(k+1+i*m1), cs.at(k), sn.at(k));

            GeneratePlaneRotation(H.at(i+i*m1), H.at(i+1+i*m1), cs.at(i), sn.at(i));
            ApplyPlaneRotation(H.at(i+i*m1), H.at(i+1+m1*i), cs.at(i), sn.at(i));
            ApplyPlaneRotation(s.at(i), s.at(i+1), cs.at(i), sn.at(i));

            if ((resid = fabs(s.at(i+1)) / normb) < tol) {
                UpdateSolution(x, i, H, s, v);
                tol = resid;
                max_iter = j;
                return 0;
            }
        }
        UpdateSolution(x, i - 1, H, s, v);
        r = b - A * x;
        beta = Norm(r);
        if ((resid = beta / normb) < tol) {
            tol = resid;
            max_iter = j;
            return 0;
        }
    }
//    std::cout<<"gmres failed, error="<<resid<<"\n";
    tol = resid;
    return 1;
}



#endif // GMRES_H
