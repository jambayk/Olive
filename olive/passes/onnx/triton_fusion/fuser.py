# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Tuple, Union

from olive.passes.onnx.triton_fusion.codegen.ops import ELEMENTWISE_OPS, ELEMENTWISE_TWO_INPUT_OPS
from olive.passes.onnx.triton_fusion.codegen.triton_generator import create_kernel


class Fuser:
    def __init__(self, base_op: str, dtype: str):
        assert self.is_valid_base_op(base_op), f"Unsupported base op: {base_op}"
        self.base_op = base_op
        self.dtype = dtype
        self.fused_ops = []

    @classmethod
    def is_valid_base_op(cls, op):
        return op == "MatMul" or op in ELEMENTWISE_OPS or op in ELEMENTWISE_TWO_INPUT_OPS

    @classmethod
    def is_valid_fused_op(cls, op):
        return op in ELEMENTWISE_OPS or op in ELEMENTWISE_TWO_INPUT_OPS

    def add_fused_op(self, op):
        assert self.is_valid_fused_op(op), f"Unsupported fused op: {op}"
        self.fused_ops.append(op)

    def write_triton_kernel(self, out_dir: Union[str, Path]) -> Tuple[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        kernel_name, kernel_code = create_kernel(self.base_op, self.fused_ops, self.dtype)
        with Path(out_dir / f"{kernel_name}.py").open("w") as f:
            f.write(kernel_code)

        return kernel_name, out_dir / f"{kernel_name}.py"
