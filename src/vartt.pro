TEMPLATE = lib
CONFIG = staticlib c++14


SOURCES +=\
    utils.cpp \
    tensor.cpp \
    index.cpp \
    mps.cpp \
    dmrg_gs.cpp \
    supertensor.cpp \
    dmrg_se_gs.cpp \
    dmrg_oe_gs.cpp \
    dmrg_wse_gs.cpp \
    parameters.cpp

HEADERS += \
    tensor.h \
    utils.h \
    index.h \
    mps.h \
    superblock.h \
    dmrg_gs.h \
    supertensor.h \
    lanczos.h \
    dmrg_se_gs.h \
    dmrg_oe_gs.h \
    dmrg_res_gs.h \
    dmrg_wse_gs.h \
    dmrg1_gs.h \
    dmrg1_wse_gs.h \
    dmrg1_res_gs.h \
    dmrg_res_cv.h \
    correctionvector.h \
    gmres_m.h \
    parameters.h \
    dmrg_0_jd_gs.h

