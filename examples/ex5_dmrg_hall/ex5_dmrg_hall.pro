CONFIG -= qt


INCLUDEPATH += ../../src
LIBS += -L../../src -lvartt
LIBS += -llapacke -lopenblas -larmadillo -lfftw3

INCLUDEPATH += /home/yurielnf/lib/spectra/include
INCLUDEPATH += /usr/local/include/eigen3

SOURCES +=\
    freefermions.cpp \
    main.cpp

HEADERS += \
    freefermions.h \
    hamhall.h


QMAKE_CFLAGS+=-pg
QMAKE_CXXFLAGS+=-pg
QMAKE_LFLAGS+=-pg
