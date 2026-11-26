cmake_minimum_required(VERSION 3.20)
project(poisson LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS C)   # Nebula's .mat material files & stage cache .h5 I/O
find_package(Python3 COMPONENTS Interpreter)

# ---------------------------------------------------------------- Vendored BLAKE3 (v1.8.7)
set(BLAKE3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/blake3")
set(BLAKE3_SOURCES
    "${BLAKE3_DIR}/blake3.c"
    "${BLAKE3_DIR}/blake3_dispatch.c"
    "${BLAKE3_DIR}/blake3_portable.c"
)
set(BLAKE3_DEFS "")

if(UNIX AND NOT APPLE AND (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$"))
    enable_language(ASM)
    list(APPEND BLAKE3_SOURCES
        "${BLAKE3_DIR}/blake3_sse2_x86-64_unix.S"
        "${BLAKE3_DIR}/blake3_sse41_x86-64_unix.S"
        "${BLAKE3_DIR}/blake3_avx2_x86-64_unix.S"
        "${BLAKE3_DIR}/blake3_avx512_x86-64_unix.S"
    )
elseif((CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
       AND (NOT DEFINED CMAKE_C_BYTE_ORDER OR CMAKE_C_BYTE_ORDER STREQUAL "LITTLE_ENDIAN"))
    list(APPEND BLAKE3_SOURCES "${BLAKE3_DIR}/blake3_neon.c")
    list(APPEND BLAKE3_DEFS BLAKE3_USE_NEON=1)
else()
    list(APPEND BLAKE3_DEFS
        BLAKE3_NO_SSE2
        BLAKE3_NO_SSE41
        BLAKE3_NO_AVX2
        BLAKE3_NO_AVX512
        BLAKE3_USE_NEON=0
    )
endif()

add_library(blake3 STATIC ${BLAKE3_SOURCES})
set_target_properties(blake3 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(blake3 PUBLIC "${BLAKE3_DIR}")
target_compile_definitions(blake3 PRIVATE ${BLAKE3_DEFS})

# Shared library loaded by Python (poisson_runner.hashing) via ctypes for full-speed BLAKE3.
add_library(poisson_blake3 SHARED src/blake3/blake3_hash.cpp)
target_include_directories(poisson_blake3 PUBLIC include)
target_link_libraries(poisson_blake3 PRIVATE blake3)
target_compile_options(poisson_blake3 PRIVATE -Wall -Wextra -Wpedantic)

# Static BLAKE3 wrapper for C++ tests/tools that need hashing without coupling core libraries.
add_library(poisson_blake3_static STATIC src/blake3/blake3_hash.cpp)
target_include_directories(poisson_blake3_static PUBLIC include)
target_link_libraries(poisson_blake3_static PRIVATE blake3)
target_compile_options(poisson_blake3_static PRIVATE -Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------- Core C++ stage libraries
# Core libraries stay free of caching, BLAKE3, and stage-I/O concerns.
add_library(poisson_common STATIC
    src/common/quadrature.cpp
    src/common/mesh_locator.cpp
)
target_include_directories(poisson_common PUBLIC include)
target_compile_options(poisson_common PRIVATE -Wall -Wextra -Wpedantic)

add_library(poisson_importer STATIC
    src/importer/importer.cpp
)
target_include_directories(poisson_importer PUBLIC include)
target_include_directories(poisson_importer PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(poisson_importer PRIVATE ${HDF5_LIBRARIES})
target_compile_options(poisson_importer PRIVATE -Wall -Wextra -Wpedantic)

add_library(poisson_amr STATIC
    src/amr/estimator_geometry.cpp
    src/amr/error_estimator.cpp
    src/amr/residual_estimators.cpp
    src/amr/recovery_estimator.cpp
    src/amr/trajectory_estimator.cpp
    src/amr/marking.cpp
    src/amr/refinement_common.cpp
    src/amr/refiner.cpp
    src/amr/longest_edge_bisection.cpp
    src/amr/red_green_refinement.cpp
    src/amr/mesh_adapter.cpp
    src/amr/error_based_adapter.cpp
)
target_include_directories(poisson_amr PUBLIC include)
target_link_libraries(poisson_amr PUBLIC poisson_common)
target_compile_options(poisson_amr PRIVATE -Wall -Wextra -Wpedantic)

# Verification cases: exact solutions and the input files of their problems.
add_library(poisson_verification STATIC
    src/verification/case_support.cpp
    src/verification/verification_case.cpp
    src/verification/layered_cases.cpp
    src/verification/image_cases.cpp
    src/verification/case_inputs.cpp
)
target_include_directories(poisson_verification PUBLIC include)
target_link_libraries(poisson_verification PRIVATE Eigen3::Eigen)
target_compile_options(poisson_verification PRIVATE -Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------- Core C++ umbrella library
add_library(poisson INTERFACE)
target_link_libraries(poisson INTERFACE poisson_common poisson_importer poisson_amr poisson_verification Eigen3::Eigen)

# ---------------------------------------------------------------- Stage HDF5/JSON serialization library
# Split into one translation unit per data type so each stage CLI binary only pulls in the
# serializers it actually calls (editing SolverResult/AssembledProblem I/O does not change
# poisson_import or poisson_adapt's compiled binary hash).
add_library(poisson_io STATIC
    src/io/h5_common.cpp
    src/io/mesh_io.cpp
    src/io/free_charges_io.cpp
    src/io/assembled_problem_io.cpp
    src/io/solver_result_io.cpp
    src/io/amr_io.cpp
    src/io/verification_io.cpp
)
target_include_directories(poisson_io PUBLIC include)
target_include_directories(poisson_io PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(poisson_io PUBLIC Eigen3::Eigen PRIVATE ${HDF5_LIBRARIES})
target_compile_options(poisson_io PRIVATE -Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------- Per-stage CLI binaries
# Each stage variant lives in its own subfolder under cli/<stage>/<variant>.cpp
# and builds as an independent executable.
add_executable(poisson_import cli/importer/poisson_import.cpp)
target_include_directories(poisson_import PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(poisson_import PRIVATE poisson_importer poisson_io)
target_compile_options(poisson_import PRIVATE -Wall -Wextra -Wpedantic)

add_executable(poisson_adapt cli/amr/poisson_adapt.cpp)
target_include_directories(poisson_adapt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(poisson_adapt PRIVATE poisson_amr poisson_io)
target_compile_options(poisson_adapt PRIVATE -Wall -Wextra -Wpedantic)

add_executable(poisson_case cli/verification/poisson_case.cpp)
target_include_directories(poisson_case PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(poisson_case PRIVATE poisson_verification poisson_io)
target_compile_options(poisson_case PRIVATE -Wall -Wextra -Wpedantic)

# ---------------------------------------------------------------- Tests
enable_testing()

add_executable(test_importer tests/test_importer.cpp)
target_link_libraries(test_importer PRIVATE poisson ${HDF5_LIBRARIES})
target_include_directories(test_importer PRIVATE ${HDF5_INCLUDE_DIRS})
target_compile_options(test_importer PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME importer COMMAND test_importer)

foreach(name constants quadrature locator marking estimators refiners adapter)
    add_executable(test_${name} tests/test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE poisson)
    target_compile_options(test_${name} PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME ${name} COMMAND test_${name})
endforeach()

add_executable(test_verification tests/test_verification.cpp)
target_link_libraries(test_verification PRIVATE poisson)
target_compile_options(test_verification PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME verification COMMAND test_verification)

add_executable(test_serialization tests/test_serialization.cpp)
target_link_libraries(test_serialization PRIVATE poisson poisson_io poisson_blake3_static ${HDF5_LIBRARIES})
target_include_directories(test_serialization PRIVATE ${HDF5_INCLUDE_DIRS})
target_compile_options(test_serialization PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME serialization COMMAND test_serialization)

add_executable(io_roundtrip_tool tests/io_roundtrip_tool.cpp)
target_link_libraries(io_roundtrip_tool PRIVATE poisson_io poisson_common)
target_compile_options(io_roundtrip_tool PRIVATE -Wall -Wextra -Wpedantic)

if(Python3_FOUND)
    add_test(
        NAME runner
        COMMAND ${Python3_EXECUTABLE} -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -p "test_*.py" -v
    )
    set_tests_properties(runner PROPERTIES
        ENVIRONMENT "POISSON_BUILD_DIR=$<TARGET_FILE_DIR:poisson_import>;PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
    )
endif()
