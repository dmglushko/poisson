cmake_minimum_required(VERSION 3.20)
project(poisson CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(HDF5 REQUIRED COMPONENTS C)   # Nebula's .mat material files

add_library(poisson src/importer.cpp)
target_include_directories(poisson PUBLIC include)
target_include_directories(poisson PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(poisson PRIVATE ${HDF5_LIBRARIES})
target_compile_options(poisson PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(test_importer tests/test_importer.cpp)
target_link_libraries(test_importer PRIVATE poisson ${HDF5_LIBRARIES})
target_include_directories(test_importer PRIVATE ${HDF5_INCLUDE_DIRS})
target_compile_options(test_importer PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME importer COMMAND test_importer)
