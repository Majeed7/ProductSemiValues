cmake_minimum_required(VERSION 3.15)
project(pgshapley_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

# Get NumPy include directory via Python
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_RESULT
)

if(NOT NUMPY_RESULT EQUAL 0)
    # Fallback: try PYTHON_EXECUTABLE (set by pybind11)
    execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NUMPY_RESULT2
    )
    if(NOT NUMPY_RESULT2 EQUAL 0)
        message(FATAL_ERROR "Could not find NumPy include directory")
    endif()
endif()

pybind11_add_module(_core csrc/explain_kernel.cpp)
target_include_directories(_core PRIVATE ${NUMPY_INCLUDE_DIR})

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_core PRIVATE -O3 -ffast-math -march=native -g)
elseif(MSVC)
    target_compile_options(_core PRIVATE /O2 /fp:fast)
endif()

install(TARGETS _core DESTINATION pgshapley)
