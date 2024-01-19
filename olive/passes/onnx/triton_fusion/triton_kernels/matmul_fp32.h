#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

CUresult matmul_fp32_128x256x64x8_warps8xstages3(CUstream stream, CUdeviceptr a_ptr, CUdeviceptr b_ptr, CUdeviceptr c_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int32_t stride_cm, int32_t stride_cn);
void load_matmul_fp32_128x256x64x8_warps8xstages3();
void unload_matmul_fp32_128x256x64x8_warps8xstages3();
    
int matmul_fp32_get_num_algos(void);

CUresult matmul_fp32_default(CUstream stream, CUdeviceptr a_ptr, CUdeviceptr b_ptr, CUdeviceptr c_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int32_t stride_cm, int32_t stride_cn);
CUresult matmul_fp32(CUstream stream, CUdeviceptr a_ptr, CUdeviceptr b_ptr, CUdeviceptr c_ptr, int32_t M, int32_t N, int32_t K, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn, int32_t stride_cm, int32_t stride_cn, int algo_id);
void load_matmul_fp32();
void unload_matmul_fp32();

#ifdef __cplusplus
}
#endif