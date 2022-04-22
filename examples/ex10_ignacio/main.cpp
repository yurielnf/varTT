#include <iostream>
#include"dmrg_res_gs.h"
#include"dmrg1_wse_gs.h"
#include"dmrg_krylov_gs.h"
#include"dmrg_res_cv.h"
#include"tensor.h"
#include"parameters.h"

#include<armadillo>

using namespace std;
using namespace arma;

///////////////////////////////Parameters2Nacho///////////////////////////////////
class Parameters2
{
public:
    Parameters2() {};
    void ReadParameters2(string filename)
    {
        ifstream in(filename);
        if (!in.is_open())
            throw invalid_argument("I couldn't open parameter file");
        string param;
        double rd;
        string rs;
        int ord;
        bool dON;
        while (!in.eof())
        {
            in >> param;
            if (param == "U") {
                in >> rd;
                U = rd;
            }
            if (param == "t") {
                in >> rd;
                t = rd;
            }
            if (param == "space") {
                in >> rs;
                space = rs;
            }
            if (param == "boundary") {
                in >> rs;
                boundary = rs;
            }
            if (param == "mu") {
                in >> rd;
                mu = rd;
            }
            if (param == "order") {
                in >> ord;
                order = ord;
            }
            if (param == "GetON") {
                in >> dON;
                ON = dON;
            }
            if (param == "lx") {
                in >> ord;
                lx = ord;
            }
            if (param == "ly") {
                in >> ord;
                ly = ord;
            }


        }
    }
    double sU() const { return U; }
    double st() const { return t; }
    double smu() const { return mu; }
    string sspace() const { return space; }
    string sboundary() const { return boundary; }
    int sorder() const { return order; }
    int slx() const { return lx; }
    int sly() const { return ly; }
    void set_space(string s)
    {
        space = s;
    }
    void set_order(int o)
    {
        order = o;
    }
    bool sON() const { return ON; }
private:
    double U;
    double t;
    string space;
    string boundary;
    double mu;
    int order;
    bool ON;
    int lx;
    int ly;

};
/////////////////////////////////////////WAVELETS/////////////////////////////////
int ind(int i, int j, int Lx)
{
    return i * Lx + j;
}
mat Daub4(int d)
{
    double c0 = (1 + sqrt(3)) / (4 * sqrt(2));
    double c1 = (3 + sqrt(3)) / (4 * sqrt(2));
    double c2 = (3 - sqrt(3)) / (4 * sqrt(2));
    double c3 = (1 - sqrt(3)) / (4 * sqrt(2));
    vec coef = { c0,c1,c2,c3 };
    mat d4(d, d, fill::zeros);
    int s = 0;
    for (int i = 0; i < d; i += 2)
    {
        for (int j = 0; j < 4; j++)
        {
            d4(i % d, (j + s) % d) = coef(j);
        }
        s += 2;
    }
    s = 0;
    for (int i = 1; i < d; i += 2)
    {
        for (int j = 0; j < 4; j++)
        {
            d4(i % d, (j + s) % d) = pow(-1, j) * coef(4 - 1 - j);
        }
        s += 2;
    }

    return d4;
}
vec invWavelet(vec S) //Transformacion inversa a un vector
{
    int d = 4;
    vec r(4);
    vec result = S;
    vec aux;
    while (d <= S.n_elem)
    {
        for (int j = 0; j < d; j++)
        {
            r[j] = result[j];
        }
        aux = r;
        for (int i = 0; i < int(d / 2); i++)
        {
            r[2 * i] = aux[i];
            r[2 * i + 1] = aux[i + int(d / 2)];
        }
        r = Daub4(d).t() * r;
        for (int j = 0; j < d; j++)
        {
            result[j] = r[j];
        }
        r.resize(int(2 * d));
        d = d * 2;
    }
    return result;
};
vec Wavelet(vec S)
{
    int d = S.n_elem;
    vec r = S;
    vec result = S;
    vec aux;
    while (d >= 4)
    {
        r = Daub4(d) * r;
        aux = r;
        for (int i = 0; i < int(d / 2); i++)
        {
            r[i] = aux[2 * i];
            r[i + int(d / 2)] = aux[2 * i + 1];
        }
        for (int j = 0; j < d; j++)
        {
            result[j] = r[j];
        }
        r.resize(int(d / 2));
        d = d / 2;
    }
    return result;
}
mat CobMatrix(int d) //Da la matriz de transformacion
{
    mat cob(d, d, fill::zeros);
    mat a = eye(d, d);
    for (int j = 0; j < d; j++)
    {
        cob.col(j) = Wavelet(a.col(j));
    }
    return cob;
}
mat Wavelet2D(mat S) //Da la matriz de transformacion
{
    mat cob(S.n_rows, S.n_cols, fill::zeros);
    for (int j = 0; j < S.n_cols; j++)
    {
        cob.col(j) = Wavelet(S.col(j));
    }
    for (int i = 0; i < S.n_rows; i++)
    {
        cob.row(i) = Wavelet(cob.row(i).t()).t();
    }
    return cob;
}
mat invWavelet2D(mat S) //Da la matriz de transformacion
{
    mat cob(S.n_rows, S.n_cols, fill::zeros);
    for (int j = 0; j < S.n_cols; j++)
    {
        cob.col(j) = invWavelet(S.col(j));
    }
    for (int i = 0; i < S.n_rows; i++)
    {
        cob.row(i) = invWavelet(cob.row(i).t()).t();
    }
    return cob;
}
vec ravel(mat X)
{
    vec sol(X.n_rows * X.n_cols, fill::zeros);
    for (int i = 0; i < X.n_rows; i++)
    {
        for (int j = 0; j < X.n_cols; j++)
        {
            sol(ind(i, j, X.n_cols)) = X(i, j);
        }
    }
    return sol;
}
mat Wavelet2DTransformationMatrix(int lx, int ly)
{
    mat T(lx * ly, lx * ly, fill::zeros);
    mat Base(lx, ly, fill::zeros);
    vec aux;
    for (int i = 0; i < lx; i++)
    {
        for (int j = 0; j < ly; j++)
        {
            Base(i, j) = 1;
            T.col(ind(i, j, ly)) = ravel(Wavelet2D(Base));
            Base(i, j) = 0;
        }
    }
    return T;
}
mat CoB2D(int lx, int ly)
{
    int L = 2 * lx * ly;
    mat T(L, L, fill::eye);
    mat cob = Wavelet2DTransformationMatrix(lx, ly);
    for (int i = 0; i < L / 2; i++)
        for (int j = 0; j < L / 2; j++)
            T(i, j) = T(L / 2 + i, L / 2 + j) = cob(i, j);
    return T;
}
mat CoB(int L, mat hop)
{
    mat T(L, L, fill::eye);
    mat cob = CobMatrix(hop.submat(0, 0, L / 2 - 1, L / 2 - 1).n_rows);
    for (int i = 0; i < L / 2; i++)
        for (int j = 0; j < L / 2; j++)
            T(i, j) = T(L / 2 + i, L / 2 + j) = cob(i, j);
    return T;
}
double CoBNN(mat T, int m, int n, int o, int p)
{
    double r = 0;
    mat Tt = T.t();
    int L = T.n_rows;
    for (int i = 0; i < L / 2; i++)
    {
        r += T(m, i) * Tt(i, n) * T(o, i + L / 2) * Tt(i + L / 2, p);
    }
    return r;
}
bool IsInGrid(int i, int j, int Lx, int Ly)
{
    if (i >= 0 && i < Lx && j >= 0 && j < Ly)
    {
        return true;
    }
    else {
        return false;
    }
}

mat LoadMat(string fname, int L)
{
    ifstream a(fname);
    mat M(L, L, fill::zeros);
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            a >> M(i, j);
        }
    }
    return M;
}
mat BasePermutation(vec a)
{
    mat rot(a.n_elem, a.n_elem, fill::zeros);
    for (int i = 0; i < a.n_elem; i++)
    {
        rot(a[i], i) = 1;
    }
    return rot;
}
mat BasePermutation(uvec a)
{
    mat rot(a.n_elem, a.n_elem, fill::zeros);

    for (int i = 0; i < a.n_elem; i++)
    {
        rot(a[i], i) = 1;
    }
    return rot;
}
vec VectorOrderHubbard(int L)
{
    vec a(L, fill::zeros);
    for (int i = 0; i < int(L / 2); i++)
    {
        a(i) = 2 * i;
    }
    for (int i = 0; i < int(L / 2); i++)
    {
        a(i + L / 2) = 2 * i + 1;
    }
    return a;
}
vec OVec(vec x, uvec ind)
{
    vec r;
    r.copy_size(x);
    for (int i = 0; i < r.size(); i++)
    {
        r(i) = x(ind(i));
    }
    return r;
}
uvec Order4(int L)
{
    uvec v(L, fill::zeros);
    for (int i = 0; i < L; i += 2)
    {
        v(int(i / 2)) = i;
    }
    for (int i = 1; i < L; i += 2)
    {
        v(L - int((i + 1) / 2)) = i;
    }
    return v;
}
//////////////////////////////////////////////////////////////////////////////////

/// L qubits, interaction U, hybridization V1
mat KinMatrixTB(int L, double U, double V1)
{
    int nbath = L / 2 - 1;
    mat kinBath(nbath, nbath, fill::zeros); // matriz de energia cinetica del banho
    for (int i = 0; i < nbath - 1; i++)
        kinBath(i, i + 1) = kinBath(i + 1, i) = pow(0.5, 0.5 * (i + 1));

    mat kin(L, L, fill::zeros); // energia cinetica total
    kin(nbath - 1, nbath) = kin(nbath, nbath - 1) = kin(nbath + 1, nbath + 2) = kin(nbath + 2, nbath + 1) = V1; // impurity at the center ---------->  oooxxooo for L=8
    for (int i = 0; i < nbath; i++)
        for (int j = 0; j < nbath; j++)
            kin(nbath - 1 - i, nbath - 1 - j) = kin(nbath + 2 + i, nbath + 2 + j) = kinBath(i, j);
    kin(L / 2 - 1, L / 2 - 1) = kin(L / 2, L / 2) = -0.5 * U;
    return kin;
}

mat Hopping(int L, double t, string boundary) // L es la cantidad de qubits de la cadena
{
    mat kin(L, L, fill::zeros);
    for (int i = 0; i < L / 2 - 1; i++) {
        kin(i, i + 1) = kin(i + 1, i) = -t; //hago la matriz de hopping 
    }
    for (int j = L / 2; j < L - 1; j++) {
        kin(j, j + 1) = kin(j + 1, j) = -t; //hago la matriz de hopping 
    }
    if (boundary == "OBC")
    {
        kin(L / 2 - 1, 0) = kin(0, L / 2 - 1) = 0;
        kin(L - 1, L / 2) = kin(L / 2, L - 1) = 0;
    }
    if (boundary == "PBC")
    {
        kin(L / 2 - 1, 0) = kin(0, L / 2 - 1) = -t;
        kin(L - 1, L / 2) = kin(L / 2, L - 1) = -t;
    }

    if (boundary == "APBC")
    {
        kin(L / 2 - 1, 0) = kin(0, L / 2 - 1) = t;
        kin(L - 1, L / 2) = kin(L / 2, L - 1) = t;
    }

    return kin;
}

mat Hopping2D(int Lx, int Ly, double t) // L es la cantidad de qubits de la cadena
{
    int L = 2 * Lx * Ly;
    mat kin(L, L, fill::zeros);
    for (int i = 0; i < Lx; i++)
    {
        for (int j = 0; j < Ly; j++)
        {
            if (IsInGrid(i, j + 1, Lx, Ly))
            {
                kin(ind(i, j, Ly), ind(i, j + 1, Ly)) = kin(ind(i, j + 1, Ly), ind(i, j, Ly)) = -t;
                kin(ind(i, j, Ly) + int(L / 2), ind(i, j + 1, Ly) + int(L / 2)) = kin(ind(i, j + 1, Ly) + int(L / 2), ind(i, j, Ly) + int(L / 2)) = -t;
            }
            if (IsInGrid(i + 1, j, Lx, Ly))
            {
                kin(ind(i, j, Ly), ind(i + 1, j, Ly)) = kin(ind(i + 1, j, Ly), ind(i, j, Ly)) = -t;
                kin(ind(i, j, Ly) + int(L / 2), ind(i + 1, j, Ly) + int(L / 2)) = kin(ind(i + 1, j, Ly) + int(L / 2), ind(i, j, Ly) + int(L / 2)) = -t;
            }
        }
    }
    return kin;
}
/*MPO HamHubbard2(int L, double U, double t)
{
    mat hop=Hopping(L,t);
    MPSSum h(1,MatSVDFixedTol(1e-12));  // writing the Hamiltonian
    for(int i=0; i<L;i++)
        for(int j=0; j<L;j++)
            if( fabs( hop(i,j) ) > 1e-13 )
                {h += Fermi(i,L,true)*Fermi(j,L,false)*hop(i,j);
        cout<<i<<" "<<j<<endl;}
    for(int k=0; k<L/2;k++)
    {
    h += Fermi(k,L,true)*Fermi(k,L,false)*Fermi(k+L/2,L,true)*Fermi(k+L/2,L,false)*U;
    cout<<k<<" "<<k+L/2<<endl;
    }
    return h.toMPS().Sweep();
}*/

MPO HamHubbard(int L, double U, double t, string space, string boundary, double mu, int ord)
{
    mat hop = Hopping(L, t, boundary);
    mat T;
    mat B(L, L, fill::eye);
    if (ord == 2)
    {
        B = BasePermutation(VectorOrderHubbard(L));
    }
    if (space == "Wavelet")
    {
        T = B * CoB(L, hop);
    }
    if (space == "Real")
    {
        T = B * eye(L, L);
    }
    if (space == "ON")
    {
        mat M = LoadMat("T_OrbitalNaturals_L=" + to_string(L) + ".dat", L);
        mat eigvec;
        vec eigval;
        eig_sym(eigval, eigvec, M);
        vec NOT = eigval - 0.5;
        uvec matrix1 = sort_index(abs(NOT));
        uvec Ord = Order4(L);
        if (ord == 3)
        {
            B = BasePermutation(matrix1);
        }
        if (ord == 4)
        {
            B = BasePermutation(matrix1) * BasePermutation(Ord);
        }
        T = B.t() * eigvec.t();
    }

    mat kin = T * hop * T.t();
    MPSSum h(1, MatSVDFixedTol(1e-12));  // writing the Hamiltonian
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            if (fabs(kin(i, j)) > 1e-13)
            {
                h += Fermi(i, L, true) * Fermi(j, L, false) * kin(i, j);
            }
//    hop.print("Matriz de hopping real:");
//    kin.print("Matriz de hopping transformada:");
//    T.print("Matriz de cambio de base:");
    if (ord == 2 && space == "Wavelet")
    {
        for (int m = 0; m < L; m++)
        {
            for (int n = 0; n < L; n++)
            {
                for (int o = 0; o < L; o++)
                {
                    for (int p = 0; p < L; p++) {
                        double ut = CoBNN(T, m, n, o, p);
                        h += Fermi(m, L, true) * Fermi(n, L, false) * Fermi(o, L, true) * Fermi(p, L, false) * ut * U;
                    }
                }
            }
        }
    }
    if (space == "ON")
    {
        for (int m = 0; m < L; m++)
        {
            for (int n = 0; n < L; n++)
            {
                for (int o = 0; o < L; o++)
                {
                    for (int p = 0; p < L; p++) {
                        double ut = CoBNN(T, m, n, o, p);
                        h += Fermi(m, L, true) * Fermi(n, L, false) * Fermi(o, L, true) * Fermi(p, L, false) * ut * U;
                    }
                }
            }
        }
    }

    if (ord == 2 && space == "Real")
    {
        for (int k = 0; k < L / 2; k++)
        {
            h += Fermi(2 * k, L, true) * Fermi(2 * k, L, false) * Fermi(2 * k + 1, L, true) * Fermi(2 * k + 1, L, false) * U;
            cout << 2 * k << 2 * k + 1 << endl;
        }
    }

    if (ord == 1 && space == "Real")
    {
        for (int k = 0; k < L / 2; k++)
        {
            h += Fermi(k, L, true) * Fermi(k, L, false) * Fermi(k + L / 2, L, true) * Fermi(k + L / 2, L, false) * U;
        }
    }

    if (ord == 1 && space == "Wavelet")
    {
        for (int m = 0; m < L / 2; m++)
        {
            for (int n = 0; n < L / 2; n++)
            {
                for (int o = 0; o < L / 2; o++)
                {
                    for (int p = 0; p < L / 2; p++) {
                        double ut = CoBNN(T, m, n, o + L / 2, p + L / 2);
                        if (abs(ut)<1e-5) continue;
                        h += Fermi(m, L, true) * Fermi(n, L, false) * Fermi(o + L / 2, L, true) * Fermi(p + L / 2, L, false) * ut * U;
                    }
                }
            }
        }
    }



    for (int i = 0; i < L; i++) {
        h += Fermi(i, L, true) * Fermi(i, L, false) * (-mu);
    }

    return h.toMPS().Sweep();
}
mat Transformation(int lx,int ly, string space, int ord)
{
    int L = 2 * lx * ly;
    mat T(L, L, fill::eye);
    mat B=BasePermutation(VectorOrderHubbard(L));
    if (space == "Real" && ord == 1)
    {
        return T;
    }
    if (space == "Real" && ord == 2)
    {
        return B;
    }
    if (space == "Wavelet" && ord == 1)
    {
        T = CoB2D(lx, ly);
        return T;
    }
    if (space == "Wavelet" && ord == 2)
    {
        T = CoB2D(lx, ly);
        return B*T;
    }
    if (space == "ON" && ord == 1)
    {
        mat M = LoadMat("T_OrbitalNaturals_L=" + to_string(L) + ".dat", L);
        mat eigvec;
        vec eigval;
        eig_sym(eigval, eigvec, M);
        T = eigvec.t();
        return T;
    }
    if (space == "ON" && ord == 3)
    {
        mat M = LoadMat("T_OrbitalNaturals_L=" + to_string(L) + ".dat", L);
        mat eigvec;
        vec eigval;
        eig_sym(eigval, eigvec, M);
        vec NOT = eigval - 0.5;
        uvec matrix1 = sort_index(abs(NOT));
        uvec Ord = Order4(L);
        B = BasePermutation(matrix1);
        T = B.t() * eigvec.t();
        return T;
    }
    if (space == "ON" && ord == 4)
    {
        mat M = LoadMat("T_OrbitalNaturals_L=" + to_string(L) + ".dat", L);
        mat eigvec;
        vec eigval;
        eig_sym(eigval, eigvec, M);
        vec NOT = eigval - 0.5;
        uvec matrix1 = sort_index(abs(NOT));
        uvec Ord = Order4(L);
        B = BasePermutation(matrix1) * BasePermutation(Ord);
        T = B.t() * eigvec.t();
        return T;
    }
    return T;
}

MPO HamHubbard2D(int Lx, int Ly, double U, double t, string space, string boundary, double mu, int ord)
{
    cout<<"hola"<<endl;
    mat hop = Hopping2D(Lx, Ly, t);
    //hop.print("Matriz de hopping real:");
    int L = 2 * Lx * Ly;
    mat T = Transformation(Lx,Ly, space, ord);
    //T.print("Matriz de transformacion:");
    mat kin = T*hop*T.t();
    //kin.print("Matriz de hopping transformada:");
    MPSSum h(1, MatSVDFixedTol(1e-12));  // writing the Hamiltonian
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            if (fabs(kin(i, j)) > 1e-13)
            {
                h += Fermi(i, L, true) * Fermi(j, L, false) * kin(i, j);
            }
    if (ord == 1 && space == "Real")
    {
        for (int i = 0; i < L / 2; i++) {
            h += Fermi(i, L, true) * Fermi(i, L, false) * Fermi(i + L / 2, L, true) * Fermi(i + L / 2, L, false) * U;
        }
    }
    else
    {
        for (int m = 0; m < L; m++)
        {
            for (int n = 0; n < L; n++)
            {
                for (int o = 0; o < L; o++)
                {
                    for (int p = 0; p < L; p++) {
                        double ut = CoBNN(T, m, n, o, p);
                        h += Fermi(m, L, true) * Fermi(n, L, false) * Fermi(o, L, true) * Fermi(p, L, false) * ut * U;
                    }
                }
            }
        }
    }
  
    for (int i = 0; i < L; i++) {
        h += Fermi(i, L, true) * Fermi(i, L, false) * (-mu);
    }

    return h.toMPS().Sweep();
}

/// L qubits, interaction U, hybridization V1
MPO HamSiamTB(int L, double U, double V1)
{
    mat kin = KinMatrixTB(L, U, V1);

    MPSSum h(1, MatSVDFixedTol(1e-12));  // writing the Hamiltonian
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            if (fabs(kin(i, j)) > 1e-13)
                h += Fermi(i, L, true) * Fermi(j, L, false) * kin(i, j);

    h += Fermi(L / 2 - 1, L, true) * Fermi(L / 2 - 1, L, false) * Fermi(L / 2, L, true) * Fermi(L / 2, L, false) * U;
    return h.toMPS().Sweep();
}


/// L qubits, interaction U, hybridization V1
MPO HamSiamTBStar(int L, double U, double V1)
{
    mat k = KinMatrixTB(L, U, V1);
    vec eigval;
    mat eigvec;
    //k.submat(0,0,L/2-2,L/2-2).print("bath=");
    eig_sym(eigval, eigvec, k.submat(0, 0, L / 2 - 2, L / 2 - 2));

    mat rot(k.n_rows, k.n_cols, fill::eye);
    for (int i = 0; i < L / 2 - 1; i++)
        for (int j = 0; j < L / 2 - 1; j++)
            rot(i, j) = rot(L - 1 - i, L - 1 - j) = eigvec(i, j);               // configuration oooxxooo for L=8

    mat kin = rot.t() * k * rot;

    MPSSum h(1, MatSVDFixedTol(1e-12));  // writing the Hamiltonian
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            if (fabs(kin(i, j)) > 1e-13)
                h += Fermi(i, L, true) * Fermi(j, L, false) * kin(i, j);

    h += Fermi(L / 2 - 1, L, true) * Fermi(L / 2 - 1, L, false) * Fermi(L / 2, L, true) * Fermi(L / 2, L, false) * U;
    return h.toMPS().Sweep();
}

/// L qubits, interaction U, hybridization V1
MPO HamSiamTBWavelet(int L, double U, double V1)
{
    mat k = KinMatrixTB(L, U, V1);
    mat rot(k.n_rows, k.n_cols, fill::eye);
    mat cob = CobMatrix(k.submat(0, 0, L / 2 - 2, L / 2 - 2).n_rows);
    for (int i = 0; i < L / 2 - 1; i++)
        for (int j = 0; j < L / 2 - 1; j++)
            rot(i, j) = rot(L - 1 - i, L - 1 - j) = cob(i, j);//rot(L/2-2-i,L/2-2-j)=rot(L/2+1+i,L/2+1+j)=cob(i,j); configuration oooxxooo for L=8


    mat kin = rot * k * rot.t();

    MPSSum h(1, MatSVDFixedTol(1e-12));  // writing the Hamiltonian
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++)
            if (fabs(kin(i, j)) > 1e-13)
                h += Fermi(i, L, true) * Fermi(j, L, false) * kin(i, j);

    h += Fermi(L / 2 - 1, L, true) * Fermi(L / 2 - 1, L, false) * Fermi(L / 2, L, true) * Fermi(L / 2, L, false) * U;
    return h.toMPS().Sweep();
}

//------------------------ Measurements -------

MPO NParticle(int L)
{
    int m = 4;
    MPSSum npart(m, MatSVDFixedTol(1e-13));
    for (int i = 0; i < L; i++)
        npart += Fermi(i, L, true) * Fermi(i, L, false);
    return npart.toMPS();
}

MPO NSite(int i, int L)
{
    int m = 4;
    MPSSum npart(m, MatSVDFixedTol(1e-13));
    npart += Fermi(i, L, true) * Fermi(i, L, false);
    return npart.toMPS();
}

MPO NOperator(int i, int j, int L)
{
    int m = 4;
    MPSSum npart(m, MatSVDFixedTol(1e-13));
    npart += Fermi(i, L, true) * Fermi(j, L, false);
    return npart.toMPS();
}



void CalculateNi(const Parameters par)
{
    MPS gs;
    gs.Load("gs.dat");
    ofstream out("ni.dat");
    int L = par.length;
    for (int i = 0; i < L; i++)
    {
        MPO rr = Fermi(i, L, true) * Fermi(i, L, false);
        double ni = Superblock({ &gs,&rr,&gs }).value();
        out << i + 1 << " " << ni << endl;
    }
}


void ExportSTable(string filename, const stdvec& qs, const TensorD& s)
{
    ofstream out(filename.c_str());
    out << "r ";
    for (auto q : qs) out << "q=" << q << " ";
    out << endl;
    for (int i = 0; i < s.dim[0]; i++)
    {
        out << i << " ";
        for (int j = 0; j<int(qs.size()); j++)
            out << s[{i, j}] << " ";
        out << endl;
    }
}

void CalculateS(const Parameters& par, const Parameters2& par2, string model)
{
    MPS gs;
    gs.Load(model + "GroundState" + string("_") + par2.sspace() + "_m=" + to_string(par.m) + "_" + par2.sboundary() + "_U=" + to_string(par2.sU()) + "_t=" + to_string(par2.st()) + "_mu=" + to_string(par2.smu()) + "_order=" + to_string(par2.sorder()) + ".dat");
    vector<double> qs = { 0.5, 1, 1.5, 2, 5, 10 };
    TensorD s({ gs.length - 1, int(qs.size()) });
    for (int i = 0; i < gs.length - 1; i++)
    {
        gs.SetPos({ i,1 });
        TensorD rho = gs.C * gs.C.t();
        TensorD eval = rho.EigenDecomposition(1).at(1);
        for (int j = 0; j<int(qs.size()); j++)
            s[{i, j}] = EntropyRenyi(eval.data(), eval.size(), qs[j]);
    }
    ExportSTable(model + "Entropy" + string("_") + par2.sspace() + "_m=" + to_string(par.m) + "_" + par2.sboundary() + "_U=" + to_string(par2.sU()) + "_t=" + to_string(par2.st()) + "_mu=" + to_string(par2.smu()) + "_order=" + to_string(par2.sorder()) + ".dat", qs, s);
}
void Calculatew(const Parameters& par, const Parameters2& par2, string model)
{
    MPS gs;
    gs.Load(model + "GroundState" + string("_") + par2.sspace() + "_m=" + to_string(par.m) + "_" + par2.sboundary() + "_U=" + to_string(par2.sU()) + "_t=" + to_string(par2.st()) + "_mu=" + to_string(par2.smu()) + "_order=" + to_string(par2.sorder()) + ".dat");
    ofstream a(model + "EigenvaluesW" + string("_") + par2.sspace() + "_m=" + to_string(par.m) + "_" + par2.sboundary() + "_U=" + to_string(par2.sU()) + "_t=" + to_string(par2.st()) + "_mu=" + to_string(par2.smu()) + "_order=" + to_string(par2.sorder()) + ".dat");
    for (int i = 0; i < gs.length - 1; i++) {
        gs.SetPos({ i,1 });
        TensorD rho = gs.C * gs.C.t();
        TensorD eval = rho.EigenDecomposition(1).at(1);
        double* evals = eval.data();
        for (int j = 0; j<int(eval.size()); j++)
        {
            a << j << " " << evals[j] << endl;
        }
        a << endl;
    }
}


//---------------------------- Test DMRG basico -------------------------------------------

void TestDMRGBasico(const Parameters& par, const Parameters2& par2, MPO op, string model)
{
    int len;
    if (model == "Hubbard")
    {
        len = par.length;
    }
    else if (model == "Hubbard2D")
    {
        len = 2 * par2.slx() * par2.sly();
    }
    else throw invalid_argument("TestDMRGBasico: unknow model");
    op.Sweep(); op.PrintSizes("HamMPO="); cout.flush();
    op.decomposer = MatQRDecomp;
    auto nop = NParticle(len);

    DMRG1_wse_gs sol(op, par.m);
    sol.tol_diag = 1e-4;
    ofstream a(model + "Energy" + string("_") + par2.sspace() + "_m=" + to_string(par.m) + "_" + par2.sboundary() + "_U=" + to_string(par2.sU()) + "_t=" + to_string(par2.st()) + "_mu=" + to_string(par2.smu()) + "_order=" + to_string(par2.sorder()) + ".dat");
    for (int k = 0; k < par.nsweep; k++)
    {
        a << "sweep " << k + 1 << " --------------------------------------\n";
        for (auto p : MPS::SweepPosSec(op.length))
        {
            sol.SetPos(p);
            sol.Solve(false);
            //sol.PrintInFile(a);

        }
        a << "nT=" << Superblock({ &sol.gs,&nop,&sol.gs }).value() << endl;
        for (int i = 0; i < len; i++)
        {
            auto nopi = NSite(i, len);
            a << i << "    " << Superblock({ &sol.gs,&nopi,&sol.gs }).value() << endl;
        }
    }
    if (par2.sspace() == "Real" && par2.sorder() == 1 && par2.sON() == 1) //Para los orbitales naturales
    {
        ofstream b("T_OrbitalNaturals_L=" + to_string(len) + ".dat");
        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < len; j++)
            {
                auto noperator = NOperator(i, j, len);
                b << Superblock({ &sol.gs,&noperator,&sol.gs }).value() << " ";
            }
            b << endl;
        }
    }
    ofstream out(model + "GroundState" + string("_") + par2.sspace() + "_m=" + to_string(par.m) + "_" + par2.sboundary() + "_U=" + to_string(par2.sU()) + "_t=" + to_string(par2.st()) + "_mu=" + to_string(par2.smu()) + "_order=" + to_string(par2.sorder()) + ".dat");
    sol.gs.Save(out);
}


int main(int argc, char* argv[])
{
    cout << "Hello World!" << endl;
    std::cout << std::setprecision(15);
    time_t t0 = time(NULL);
    srand(time(NULL));

    if (string(argv[2]) == "basic")  // ./a.out parameters.dat basic
    {
        Parameters par;
        par.ReadParameters(argv[1]);
        Parameters2 par2;
        par2.ReadParameters2(argv[3]);
        MPO hamiltonian = HamHubbard(par.length, par2.sU(), par2.st(), par2.sspace(), par2.sboundary(), par2.smu(), par2.sorder());
        cout << "hola" << endl;
        //for(int i=8;i<128;i+=8){
        //par.m=i;
        TestDMRGBasico(par, par2, hamiltonian, "Hubbard");
        CalculateS(par, par2, "Hubbard");
        Calculatew(par, par2, "Hubbard");
        //}
    }
    if (string(argv[2]) == "basic2D")  // ./a.out parameters.dat basic2D
    {
        Parameters par;
        par.ReadParameters(argv[1]);
        Parameters2 par2;
        par2.ReadParameters2(argv[3]);
        MPO hamiltonian = HamHubbard2D(par2.slx(), par2.sly(), par2.sU(), par2.st(), par2.sspace(), par2.sboundary(), par2.smu(), par2.sorder());
        for(int i=8;i<128;i+=8){
        par.m=i;
        TestDMRGBasico(par, par2, hamiltonian, "Hubbard2D");
        CalculateS(par, par2, "Hubbard2D");
        Calculatew(par, par2, "Hubbard2D");
        }
    }
    if (string(argv[2]) == "all")  // ./a.out parameters.dat all
    {
        Parameters par;
        par.ReadParameters(argv[1]);
        Parameters2 par2;
        par2.ReadParameters2(argv[3]);
        vector<string> space{ "Real","Wavelet","ON" };
        vector<vector<int>> order{ {1,2},{1,2},{1,3,4} };
        for (int i = 0; i < space.size(); i++)
        {
            par2.set_space(space[i]);
            for (int j = 0; j < order[i].size(); j++)
            {
                par2.set_order(order[i][j]);
                MPO hamiltonian = HamHubbard(par.length, par2.sU(), par2.st(), par2.sspace(), par2.sboundary(), par2.smu(), par2.sorder());
                TestDMRGBasico(par, par2, hamiltonian, "Hubbard");
                CalculateS(par, par2, "Hubbard");
                Calculatew(par, par2, "Hubbard");
            }
        }
    }
    if (string(argv[2]) == "all2D")  // ./a.out parameters.dat all2D
    {
        Parameters par;
        par.ReadParameters(argv[1]);
        Parameters2 par2;
        par2.ReadParameters2(argv[3]);
        vector<string> space{ "Real","Wavelet","ON" };
        vector<vector<int>> order{ {1,2},{1,2},{1,3,4} };
        for (int i = 0; i < space.size(); i++)
        {
            par2.set_space(space[i]);
            for (int j = 0; j < order[i].size(); j++)
            {
                par2.set_order(order[i][j]);
                MPO hamiltonian = HamHubbard2D(par2.slx(), par2.sly(), par2.sU(), par2.st(), par2.sspace(), par2.sboundary(), par2.smu(), par2.sorder());
                TestDMRGBasico(par, par2, hamiltonian, "Hubbard2D");
                CalculateS(par, par2, "Hubbard2D");
                Calculatew(par, par2, "Hubbard2D");
            }
        }
    }

    cout << "\nDone in " << difftime(time(NULL), t0) << "s" << endl;
    return 0;
}
