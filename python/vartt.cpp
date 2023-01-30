#include <carma>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "../examples/ex11_irlm/ham_irlm.h"
#include "dmrg.h"

using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;




//---------------------------------- start python module ------------------


PYBIND11_MODULE(vartt, m) {
  m.doc() = "Python interface for vartt";

  py::class_<DMRG>(m,"DMRG")
          .def(py::init<>())
          .def_readwrite("m",&DMRG::m)
          .def_readwrite("nIterMaxLanczos",&DMRG::nIterMaxLanczos)
          .def_readwrite("nsweep",&DMRG::nsweep)
          .def_readwrite("toldiag",&DMRG::toldiag)
          .def_readonly("energy",&DMRG::energy)
          .def("calculateCiCj",&DMRG::CalculateCiCj)
          .def("runDMRG0",&DMRG::runDMRG0)
          .def("runDMRG1",&DMRG::runDMRG1)
          ;

  py::class_<HamIRLM>(m,"IRLM")
          .def(py::init<arma::mat,arma::mat,double>())
          ;

  m.def("set_IRLM_Ham",[](DMRG &sol, HamIRLM const& irlm){});

}
