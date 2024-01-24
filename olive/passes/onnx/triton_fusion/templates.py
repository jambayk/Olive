# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

TL_DTYPE_MAP = {
    "fp32": "tl.float32",
    "fp16": "tl.float16",
    "int32": "tl.int32",
    "int64": "tl.int64",
    "bool": "tl.bool",
    "bf16": "tl.bfloat16",
}


matmul_template = """
import triton
import triton.language as tl

# Fused operations: {fused_ops_str}
@triton.jit
def triton_{kernel_name}(
    # pointers to matrices
    a_ptr,
    b_ptr,
    # pointer to other tensors for fused operations
    {fused_ptr_args}
    # pointer to output tensor
    y_ptr,
    # matrix dimensions
    M,
    N, 
    K, 
    # number of elements for other tensors
    {fused_numel_args}
    # attributes for fused operations
    {fused_attr_args}
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    \"\"\"Kernel for computing the matmul Y = A x B.
    A has shape (M, K), B has shape (K, N) and Y has shape (M, N)
    Matmul can be fused with other epilogue operations such as activation, bias add, etc.
    \"\"\"
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of Y it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    # -----------------------------------------------------------
    # Iterate to compute a block of the Y matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N

    # cast accumulator to destination type
    y = accumulator.to({y_dtype})

    # -----------------------------------------------------------
    # Indices for the output matrix
    offs_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_idxs = offs_ym[:, None] * N + offs_yn[None, :]
    y_mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)

    # -----------------------------------------------------------
    # Fusion with other operations
    {fused_code}

    # -----------------------------------------------------------
    # Write back the block of the output matrix Y with masks.
    y_ptrs = y_ptr + y_idxs
    tl.store(y_ptrs, y, mask=y_mask)
"""
