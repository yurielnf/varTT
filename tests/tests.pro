#TEMPLATE = app
#CONFIG += console c++11
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
    test_tensor_notation.cpp
