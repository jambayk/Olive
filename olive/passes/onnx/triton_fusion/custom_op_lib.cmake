set(custom_op_lib_include)
set(custom_op_lib_option)
set(custom_op_lib_link)
set(custom_op_lib_link_dir)

# ort related
set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include/onnxruntime)
set(ONNXRUNTIME_API_DIR ${ONNXRUNTIME_DIR}/include/onnxruntime/core/session)
list(APPEND custom_op_lib_include ${ONNXRUNTIME_INCLUDE_DIR} ${ONNXRUNTIME_API_DIR})

# triton related
# list(APPEND custom_op_lib_include ${TRITON_CUDA_DIR}/include)
# list(APPEND custom_op_lib_link_dir ${TRITON_CUDA_DIR}/lib)

# do we need to link cuda?
list(APPEND custom_op_lib_include ${CUDA_HOME}/include)
list(APPEND custom_op_lib_link_dir ${CUDA_HOME}/lib64)
list(APPEND custom_op_lib_link cuda)

set(custom_op_src
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_op/custom_op_library.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_op/custom_op_library.h
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_op/ops/fusion_ops.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_op/ops/fusion_ops.h
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_op/ops/triton_kernels/matmul_fp32.c
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_op/ops/triton_kernels/matmul_fp32.h
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_op/ops/triton_kernels/matmul_fp32.3fdb966f_01234567891011.c 
    ${CMAKE_CURRENT_SOURCE_DIR}/custom_op/ops/triton_kernels/matmul_fp32.3fdb966f_01234567891011.h
)

# print custom op src
message(STATUS "custom op src: ${custom_op_src}")

# add custom op library
add_library(custom_op_lib SHARED ${custom_op_src})
target_compile_options(custom_op_lib PRIVATE ${custom_op_lib_option})
target_include_directories(custom_op_lib PRIVATE ${custom_op_lib_include})
target_link_directories(custom_op_lib PRIVATE ${custom_op_lib_link_dir})
target_link_libraries(custom_op_lib PRIVATE ${custom_op_lib_link})

if (WIN32)
    set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-DEF:${CMAKE_CURRENT_SOURCE_DIR}/csrc/custom_op_library.def")
else()
    set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker --version-script=${CMAKE_CURRENT_SOURCE_DIR}/custom_op/custom_op_library.lds -Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
endif()
set_property(TARGET custom_op_lib APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG})