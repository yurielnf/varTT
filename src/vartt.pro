TEMPLATE = lib
CONFIG = staticlib c++14


SOURCES +=\
    utils.cpp \
    tensor.cpp \
    index.cpp \
    mps.cpp \
    dmrg_gs.cpp \
    supertensor.cpp \
    dmrg_0_gs.cpp \
    dmrg_se_gs.cpp

HEADERS += \
    tensor.h \
    utils.h \
    index.h \
    mps.h \
    superblock.h \
    dmrg_gs.h \
    supertensor.h \
    lanczos.h \
    dmrg_0_gs.h \
    dmrg_se_gs.h

