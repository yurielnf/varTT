TEMPLATE = subdirs
SUBDIRS = \
    src/vartt.pro \
    tests/tests.pro \
    examples/ex1_tensor \
    examples/ex2_mpo \
    examples/ex3_dmrg_tb

tests.depends = vartt
ex1_tensor.depends = vartt
ex2_mpo.depends = vartt
ex3_dmrg_tb.depends = vartt

DISTFILES += \
    README.md
