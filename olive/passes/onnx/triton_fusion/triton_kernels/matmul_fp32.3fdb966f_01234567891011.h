#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_matmul_fp32_3fdb966f_01234567891011(void);
void load_matmul_fp32_3fdb966f_01234567891011(void);
// tt-linker: matmul_fp32_3fdb966f_01234567891011:CUdeviceptr a_ptr, CUdeviceptr b_ptr, CUdeviceptr c_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int32_t stride_cm, int32_t stride_cn:128x256x64x8_warps8xstages3
CUresult matmul_fp32_3fdb966f_01234567891011(CUstream stream, CUdeviceptr a_ptr, CUdeviceptr b_ptr, CUdeviceptr c_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int32_t stride_cm, int32_t stride_cn);
