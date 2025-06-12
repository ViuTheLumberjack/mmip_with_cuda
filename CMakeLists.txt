include(CheckLanguage)
include(FetchContent)

cmake_minimum_required(VERSION 3.28)
project(mmip_with_cuda LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif ()

add_library(mmip_lib
        source/mmip_cpu.cpp
        source/mmip_cpu.h
        source/mmip_wrapper.cpp
        source/mmip_wrapper.h
        source/mmip_seq.cpp
        source/mmip_seq.h
)

add_executable(mmip main.cpp)
add_executable(benchmark benchmark.cpp)

target_link_libraries(mmip PUBLIC mmip_lib)
target_link_libraries(benchmark PUBLIC mmip_lib matplot)

if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    add_executable(cuda_demo cuda_demo.cu)

    add_library(mmip_lib_cuda
            source/mmip_cuda.cu
            source/mmip_cuda.cuh
    )

    target_link_libraries(mmip PUBLIC mmip_lib_cuda)
    target_link_libraries(benchmark PUBLIC mmip_lib_cuda)

    set_target_properties(cuda_demo PROPERTIES LINKER_LANGUAGE CUDA)

    set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
endif()

find_package(OpenCV REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

target_include_directories(mmip PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(benchmark PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(mmip PRIVATE
        ${OpenCV_LIBS}
        ${ImageMagick_LIBRARIES}
)
target_link_libraries(benchmark PRIVATE
        ${OpenCV_LIBS}
        ${ImageMagick_LIBRARIES}
)

find_package(OpenMP REQUIRED C CXX)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mmip_lib PRIVATE OpenMP::OpenMP_CXX)
endif ()

add_custom_target(copy_assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/images ${CMAKE_CURRENT_BINARY_DIR}/images
)

add_dependencies(mmip copy_assets)
add_dependencies(benchmark copy_assets)

if(CMAKE_CUDA_COMPILER)
    set_target_properties(cuda_demo PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 12
    )

    set_target_properties(cuda_demo PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 12
    )
endif()