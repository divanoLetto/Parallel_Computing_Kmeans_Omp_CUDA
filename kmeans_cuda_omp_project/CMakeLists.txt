project(kmeans_cuda_omp_project LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 17)
file(GLOB cuda_source_files "${CMAKE_SOURCE_DIR}/*.cu")

set(CUDA_HOST_COMPILER /usr/bin/g++-6)
find_package(CUDA QUIET REQUIRED)

set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
include_directories(/opt/local/include/libomp)
link_directories(/opt/local/lib/libomp)

find_package(PythonLibs 2.7.17)

cuda_add_executable(
        cuda_version
        main_adv.cpp
        kmeans_adv.cu
        kmeans_adv_slow.cu
        plot_adv.h
        data_generator.h
        kmeans_sequential.h
        kmeans_omp.h
        Record.h
)

set(BOOST_ROOT "/usr/lib")
find_package(Boost  1.65.1)

target_include_directories(cuda_version PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(cuda_version ${PYTHON_LIBRARIES})
target_include_directories(cuda_version PUBLIC /home/lorenzo/CLionProjects/kmeans_cuda_omp_project/matplotlib-cpp-master)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(cuda_version Threads::Threads)
