add_executable(ex5_hall
    hamhall.h
    freefermions.h 
    freefermions.cpp
    main.cpp
    
)

target_link_libraries(ex5_hall vartt)
target_link_libraries(ex5_hall fftw3)

