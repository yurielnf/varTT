CONFIG -= qt


INCLUDEPATH += ../../src
LIBS += -L../../src -lvartt
LIBS += -llapacke -lopenblas -larmadillo -lfftw3

INCLUDEPATH += /home/yurielnf/lib/spectra/include
INCLUDEPATH += /usr/local/include/eigen3

SOURCES +=\
    main.cpp

HEADERS += \
    cadenitaaa.h \
    hamnn.h
