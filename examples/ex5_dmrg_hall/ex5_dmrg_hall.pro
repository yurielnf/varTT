CONFIG -= qt


INCLUDEPATH += ../../src
LIBS += -L../../src -lvartt
LIBS += -llapacke -larmadillo -lfftw3

SOURCES +=\
    freefermions.cpp \
    main.cpp

HEADERS += \
    freefermions.h \
    hamhall.h
