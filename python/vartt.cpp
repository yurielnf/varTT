#include <carma>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "../examples/ex11_irlm/ham_irlm.h"
#include "dmrg.h"
#include "model/fermionic.h"

using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;




//---------------------------------- start python module ------------------


PYBIND11_MODULE(varttpy, m) {
  m.doc() = "Python interface for vartt";

  py::class_<TensorD>(m,"Tensor")
          .def_readonly("shape",&TensorD::dim)
          ;

  py::class_<MPS>(m,"MPS")
          .def_readonly("cores",&MPS::M)
          ;

  py::class_<DMRG_base>(m,"DMRG_base")
          .def_readwrite("bond_dim",&DMRG_base::m)
          .def_readwrite("nIter_diag",&DMRG_base::nIter_diag)
          .def_readwrite("tol_diag",&DMRG_base::tol_diag)
          .def_readwrite("use_arpack",&DMRG_base::use_arpack)
          .def("Expectation",&DMRG_base::Expectation)
          .def("correlation",&DMRG_base::correlation)
          .def("sigma",&DMRG_base::sigma)
          .def("H2",&DMRG_base::H2)
          .def_readonly("sweep",&DMRG_base::sweep)
          .def_readonly("energy",&DMRG_base::energy)
          .def_readonly("ham",&DMRG_base::ham)
          .def_readonly("gs",&DMRG_base::gs)
          ;

  py::class_<DMRG,DMRG_base>(m,"DMRG")
          .def(py::init<MPO>())
          .def(py::init<MPO,MPS>())
          .def("iterate",&DMRG::iterate)
          ;

  py::class_<DMRG0,DMRG_base>(m,"DMRG0")
          .def(py::init<MPO,int>(), "Ham"_a, "nKrylov"_a=2)
          .def(py::init<MPO,MPS,int>(), "Ham"_a, "gs"_a,"nKrylov"_a=2)
          .def("iterate",&DMRG0::iterate)
          ;

  py::class_<Fermionic>(m,"Fermionic")
          .def(py::init<arma::mat,arma::mat, std::map<std::array<int,4>,double>>(),
               "Kij"_a, "Uij"_a, "Vijkl"_a)
          .def(py::init<arma::mat,arma::mat,arma::mat>(),
               "Kij"_a, "Uij"_a, "Rot"_a)
          .def("Ham",&Fermionic::Ham, "tol"_a=1e-14)
          .def("NParticle",&Fermionic::NParticle)
          .def("CidCj",&Fermionic::CidCj)
          ;

  py::class_<HamIRLM>(m,"IRLM")
          .def(py::init<arma::mat,arma::mat,double>(),
               "tmat"_a, "Pmat"_a, "U"_a)
          .def("Ham",&HamIRLM::Ham)
          .def("NParticle",&HamIRLM::NParticle)
          .def("CalculateCiCj",&HamIRLM::CalculateCiCj)
          ;


}
