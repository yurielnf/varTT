CONFIG -= qt


INCLUDEPATH += ../../src
LIBS += -L../../src -lvartt
LIBS += -llapacke -lopenblas -larmadillo -lfftw3

SOURCES +=\
    main.cpp

HEADERS += \
    cadenitaaa.h \
    hamnn.h
