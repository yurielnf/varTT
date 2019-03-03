TEMPLATE = lib
CONFIG = staticlib c++14


SOURCES +=\
    utils.cpp \
    tensor.cpp \
    index.cpp \
    mps.cpp \
    dmrg_gs.cpp \
    supertensor.cpp

HEADERS += \
    tensor.h \
    utils.h \
    index.h \
    mps.h \
    superblock.h \
    dmrg_gs.h \
    supertensor.h \
    lanczos.h

