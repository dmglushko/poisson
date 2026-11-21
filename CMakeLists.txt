cmake_minimum_required(VERSION 3.20)
project(poisson CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS C)   # Nebula's .mat material files

add_library(poisson
    src/importer.cpp
    src/common/quadrature.cpp
    src/common/mesh_locator.cpp
    src/amr/estimator_geometry.cpp
    src/amr/error_estimator.cpp
    src/amr/residual_estimators.cpp
    src/amr/recovery_estimator.cpp
    src/amr/marking.cpp
    src/amr/refinement_common.cpp
    src/amr/refiner.cpp
    src/amr/longest_edge_bisection.cpp
    src/amr/red_green_refinement.cpp
    src/amr/mesh_adapter.cpp
    src/amr/error_based_adapter.cpp)
target_include_directories(poisson PUBLIC include)
target_include_directories(poisson PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(poisson PUBLIC Eigen3::Eigen PRIVATE ${HDF5_LIBRARIES})
target_compile_options(poisson PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(test_importer tests/test_importer.cpp)
target_link_libraries(test_importer PRIVATE poisson ${HDF5_LIBRARIES})
target_include_directories(test_importer PRIVATE ${HDF5_INCLUDE_DIRS})
target_compile_options(test_importer PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME importer COMMAND test_importer)

foreach(name quadrature locator marking estimators refiners adapter)
    add_executable(test_${name} tests/test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE poisson ${HDF5_LIBRARIES})
    target_compile_options(test_${name} PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME ${name} COMMAND test_${name})
endforeach()
