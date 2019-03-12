#TEMPLATE = app
CONFIG += console c++14
#CONFIG -= app_bundle
CONFIG -= qt

TARGET = test.x

INCLUDEPATH += ../src
LIBS += -L../src -lvartt
LIBS += -llapacke -larmadillo

SOURCES +=\
    test_tensor.cpp \
    test_main.cpp \
    test_mps.cpp \
    test_tensor_notation.cpp \
    test_superblock.cpp \
    test_dmrg_tight_binding.cpp \
    test_mpo.cpp
