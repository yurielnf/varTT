add_executable(ex11_irlm
    irlm.h
    main.cpp    
)

target_link_libraries(ex11_irlm PRIVATE vartt)
target_link_libraries(ex11_irlm PRIVATE armadillo)
target_link_libraries(ex11_irlm PRIVATE lapacke)

