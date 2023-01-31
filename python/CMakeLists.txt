pybind11_add_module(${PROJECT_NAME}py
    vartt.cpp
)

target_link_libraries(${PROJECT_NAME}py PUBLIC ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}py PUBLIC carma::carma)

target_link_libraries(${PROJECT_NAME}py PRIVATE Eigen3::Eigen)

target_link_libraries(${PROJECT_NAME}py PRIVATE lapacke)

