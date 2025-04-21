cmake_minimum_required(VERSION 3.28)
project(mmip_with_cuda)

set(CMAKE_CXX_STANDARD 20)

add_executable(mmip main.cpp
        source/mmip_cpu.cpp
        source/mmip_cpu.h
        source/mmip_wrapper.cpp
        source/mmip_wrapper.h
        source/mmip_cuda.cu
        source/mmip_cuda.cuh
        source/mmip_seq.cpp
        source/mmip_seq.h)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})
target_include_directories(mmip PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(mmip PRIVATE
        ${OpenCV_LIBS}
        ${ImageMagick_LIBRARIES}
)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mmip PRIVATE OpenMP::OpenMP_CXX)
endif ()

add_custom_target(copy_assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/images ${CMAKE_CURRENT_BINARY_DIR}/images
)
add_dependencies(mmip copy_assets)