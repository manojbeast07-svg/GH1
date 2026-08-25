find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Locate the pybind11 CMake package shipped inside the pip-installed
# pybind11 package, using the same Python interpreter CMake found above.
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_cmake_dir_result
)
if(NOT _pybind11_cmake_dir_result EQUAL 0)
    message(FATAL_ERROR "Could not locate pybind11 (pip install pybind11).")
endif()
list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmake_dir}")

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(xray_cuda
    src/bindings.cpp
    src/cuda_common.cu
    src/smoke_test.cu
    src/gpu_image.cu
    src/gaussian_basic.cu
    src/median_basic.cu
    src/sobel_basic.cu
    src/laplacian_basic.cu
    src/threshold_basic.cu
    src/pipeline_basic.cu
    src/pipeline_experimental.cu
    src/pipeline_cuda_graph.cu
    src/pipeline_cuda_graph_enhanced.cu
    src/pipeline_async.cu
    src/gaussian_enhanced.cu
    src/median_enhanced.cu
    src/sobel_enhanced.cu
    src/laplacian_enhanced.cu
    src/threshold_enhanced.cu
    src/laplacian_threshold_fused.cu
)

target_include_directories(xray_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(xray_cuda PRIVATE CUDA::cudart)

set_target_properties(xray_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Place the built extension at the project root so `import xray_cuda` works
# from there without extra path manipulation.
set_target_properties(xray_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)
foreach(_cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER ${_cfg} _cfg_upper)
    set_target_properties(xray_cuda PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${_cfg_upper} ${CMAKE_SOURCE_DIR}
        RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper} ${CMAKE_SOURCE_DIR}
    )
endforeach()
