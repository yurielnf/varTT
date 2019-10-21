TEMPLATE = subdirs
SUBDIRS = \
    src/vartt.pro \
    tests/tests.pro \
#\
#    examples/ex1_tensor \
#    examples/ex2_mpo \
#    examples/ex3_dmrg_tb \
    examples/ex4_dmrg_cv_imp\
    examples/ex5_dmrg_hall\
    examples/ex6_dmrg_cv_hernan

tests.depends = vartt
#ex1_tensor.depends = vartt
#ex2_mpo.depends = vartt
#ex3_dmrg_tb.depends = vartt
#ex4_dmrg_cv_imp.depends = vartt
ex5_dmrg_hall.depends = vartt
ex4_dmrg_cv_hernan.depends = vartt


DISTFILES += \
    README.md
