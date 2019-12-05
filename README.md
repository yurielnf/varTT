# varTT

Variational tensor train (or MPS) optimized using zero-site DMRG.
Both the Lanczos and Jacobi-Davidson corrections are implemented to avoid local minima.
Intelligent tensor contractions are supported:

```
t2("li")=t1("ijk") * t1("ljk");
```

together with the MPO automatic construction as in:

```
h += Fermi(i,L,true) * Fermi(j,L,false) * t(i,j);
h += Fermi(i,L,true) * Fermi(i,L,false) * Fermi(i+1,L,true) * Fermi(i+1,L,false);

int nsweep=8, m=200;
TypicalRunDMRG0(h,nsweep,m);
```

### External dependencies

- [armadillo](https://gitlab.com/conradsnicta/armadillo-code)
- lapacke
- blas

### Getting started

This project can be edited and build using the [`qtcreator`](https://github.com/qt-creator) IDE.


### Running the test suite

The tests are in the tests/ folder using [catch2](https://github.com/catchorg/Catch2).

### Examples

Several example codes are provided at examples/ folder.
