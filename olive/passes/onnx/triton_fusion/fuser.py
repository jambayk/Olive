# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Dict

from olive.passes.onnx.triton_fusion.codegen.ops import ELEMENTWISE_OPS, ELEMENTWISE_TWO_INPUT_OPS
from olive.passes.onnx.triton_fusion.codegen.ort_generator import create_custom_op
from olive.passes.onnx.triton_fusion.codegen.triton_generator import create_kernel
from olive.passes.onnx.triton_fusion.utils import create_custom_op_name


class Fusion:
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

    def get_triton_kernel(self) -> Dict:
        return create_kernel(self.base_op, self.fused_ops, self.dtype)

    def get_custom_op_name(self):
        return create_custom_op_name([self.base_op, *self.fused_ops], self.dtype)

    def get_custom_op(self) -> Dict:
        return create_custom_op(self.base_op, self.fused_ops, self.dtype)
