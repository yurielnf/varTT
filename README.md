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

- lapack, blas and lapacke
- python-numpy for the python library

The other dependencies are automatically downloaded by `cmake`. They are: `armadillo`, `carma`, `eigen3`, `Spectra`, `Catch2` and `pybind11`.

### Compiling the library

```
mkdir build && cd build
cmake .. -D CMAKE_BUILD_TYPE=Release
make -j4
```

### Examples

Several c++ examples are provided at `examples/` folder. For python users, there is an example in `notebook/`
