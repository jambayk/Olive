# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

elementwise_template = """
import triton
import triton.language as tl

# Fused operations: {fused_ops_str}
@triton.jit
def triton_{kernel_name}(
    # pointers to base input tensors
    a_ptr,
    {b_ptr_arg}
    # pointer to other tensors for fused operations
    {fused_ptr_args}
    # pointer to output tensor
    y_ptr,
    # number of elements for base input tensors
    a_numel,
    {b_numel_arg}
    # number of elements for other tensors
    {fused_numel_args}
    # attributes for base operation
    {base_attr_args}
    # attributes for fused operations
    {fused_attr_args}
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    \"\"\"Kernel for computing the elementwise operation Y = op(A) or Y = op(A, B).

    A has shape (a_numel,) and B has shape (b_numel,) where a_numel % b_numel == 0.
    The output Y has shape (a_numel,).
    Elementwise operation can be fused with other elementwise operations.
    \"\"\"
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of Y it should compute.
    pid = tl.program_id(axis=0).to(tl.int64)
    y_idxs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    y_mask = y_idxs < a_numel

    # -----------------------------------------------------------
    # Load the base input tensors.
    a = tl.load(a_ptr + y_idxs, mask=y_mask)
    {b_load_code}

    # -----------------------------------------------------------
    # Perform the base operation.
    {base_code}

    # -----------------------------------------------------------
    # Fusion with other operations
    {fused_code}

    # -----------------------------------------------------------
    # Write back the output tensor Y with masks.
    y_ptrs = y_ptr + y_idxs
    tl.store(y_ptrs, y, mask=y_mask)
"""

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

    A has shape (M, K), B has shape (K, N) and Y has shape (M, N).
    Matmul can be fused with elementwise operations such as bias addition, activation, etc.
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
