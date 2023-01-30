set(SOURCES
    utils.cpp 
    tensor.cpp 
    index.cpp 
    mps.cpp 
    dmrg_gs.cpp 
    supertensor.cpp 
    dmrg_se_gs.cpp 
    dmrg_oe_gs.cpp 
    dmrg_wse_gs.cpp 
    parameters.cpp 
    spectrawrapper.cpp
)

set(HEADERS
    dmrg.h
    tensor.h 
    utils.h 
    index.h 
    mps.h 
    superblock.h 
    dmrg_gs.h 
    supertensor.h 
    lanczos.h 
    dmrg_se_gs.h 
    dmrg_oe_gs.h 
    dmrg_res_gs.h 
    dmrg_wse_gs.h 
    dmrg1_gs.h 
    dmrg1_wse_gs.h 
    dmrg1_res_gs.h 
    dmrg_res_cv.h 
    correctionvector.h 
    gmres_m.h 
    parameters.h 
    dmrg_jd_gs.h 
    dmrg1_jd_gs.h 
    dmrg_krylov_gs.h 
    cg.h 
    gmres.h 
    dmrg_jacobi_davidson_gs.h 
    lanczoscv.h 
    spectrawrapper.h
)

add_library(${PROJECT_NAME} 
    ${HEADERS}
    ${SOURCES}
)

#---------- external dependencies ---------

#find_package(Armadillo REQUIRED)
#target_include_directories(${PROJECT_NAME} PRIVATE ${ARMADILLO_INCLUDE_DIRS})
#target_link_libraries(${PROJECT_NAME} PRIVATE ${ARMADILLO_LIBRARIES})

target_link_libraries(${PROJECT_NAME} PRIVATE armadillo)

#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries (${PROJECT_NAME} PRIVATE Eigen3::Eigen)

target_link_libraries(${PROJECT_NAME} PRIVATE Spectra)

target_link_libraries(${PROJECT_NAME} PRIVATE lapacke)


#LIBS += -llapacke -lopenblas -larmadillo -lfftw3

#INCLUDEPATH += /home/yurielnf/lib/spectra/include
#INCLUDEPATH += /usr/include/eigen3
