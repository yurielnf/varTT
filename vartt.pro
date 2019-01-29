TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    tests/test_tensor.cpp \
    utils.cpp \
    tensor.cpp

HEADERS += \
    tensor.h \
    utils.h


LIBS += -llapacke -larmadillo
