TEMPLATE = subdirs
SUBDIRS = \
    src/vartt.pro \
    tests/tests.pro \
    examples/ex1_tensor \
    examples/ex2_mpo

tests.depends = vartt
ex1_tensor.depends = vartt
ex2_tensor.depends = vartt

DISTFILES += \
    README.md
