cmake_minimum_required(VERSION 3.9)

project(vartt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(external/external.cmake)

add_subdirectory(src)

include_directories(${PROJECT_NAME} PRIVATE src)

add_subdirectory(python)


#add_subdirectory(examples/ex10_ignacio)
#add_subdirectory(examples/ex5_dmrg_hall)
add_subdirectory(examples/ex11_irlm)

#set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
