# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import List, Tuple, Union

from olive.common.config_utils import ConfigBase

# TODO(jambayk): Create a class to hold all ops and their metadata, perform transformations, etc.


# -----------------------------------------------------------
# Elementwise operations
# -----------------------------------------------------------


# op_name: (attributes, number of temporary variables, template for triton representation)
class ElementwiseOp(ConfigBase):
    attributes: List[Tuple[str, str]] = None
    num_temp_vars: int = 0
    triton_template: Union[str, List[str]]


# elementwise operations that have a single input
elementwise_ops = {
    # Math functions
    "Abs": ElementwiseOp(triton_template="tl.math.abs({in0})"),
    "Ceil": ElementwiseOp(triton_template="tl.math.ceil({in0})"),
    "Erf": ElementwiseOp(triton_template="tl.math.erf({in0})"),
    "Exp": ElementwiseOp(triton_template="tl.exp({in0})"),
    "Floor": ElementwiseOp(triton_template="tl.math.floor({in0})"),
    "Log": ElementwiseOp(triton_template="tl.log({in0})"),
    # Activation functions
    "Celu": ElementwiseOp(
        attributes=[("alpha", "fp32")],
        triton_template="tl.where({in0} > 0.0, {in0}, {alpha} * (tl.exp({in0} / {alpha}) - 1.0))",
    ),
    "Elu": ElementwiseOp(
        attributes=[("alpha", "fp32")], triton_template="tl.where({in0} > 0.0, {in0}, {alpha} * (tl.exp({in0}) - 1.0))"
    ),
    "LeakyRelu": ElementwiseOp(
        attributes=[("alpha", "fp32")], triton_template="tl.where({in0} > 0.0, {in0}, {alpha} * {in0})"
    ),
    "Relu": ElementwiseOp(triton_template="tl.where({in0} > 0, {in0}, 0.0)"),
    "Selu": ElementwiseOp(
        attributes=[("alpha", "fp32"), ("gamma", "fp32")],
        triton_template="tl.where({in0} > 0.0, {gamma} * {in0}, {gamma} * ({alpha} * tl.exp({in0}) - {alpha}))",
    ),
    # TODO(jambayk): Investigate support for string attributes like "approximation" for Gelu
    "HardSigmoid": ElementwiseOp(
        attributes=[("alpha", "fp32"), ("beta", "fp32")],
        triton_template="tl.where({in0} < -{beta}, 0.0, tl.where({in0} > {beta}, 1.0, {alpha} * {in0} + {beta}))",
    ),
    "Sigmoid": ElementwiseOp(triton_template="tl.sigmoid({in0})"),
    # sign functions
    "Neg": ElementwiseOp(triton_template="-{in0}"),
    "Not": ElementwiseOp(triton_template="~{in0}"),
}

# elementwise operations that have two inputs
# broadcasting support is currently limited to the following unidirectional constraints:
# - shape of second input must be a suffix of the shape of the first input
# - Only leading 1s are allowed in the shape of the second input
# - Example [2, 3, 4, 5]: [1], [5], [1, 5], [4, 5], ...
# TODO(jambayk): Add support for multidiensional broadcasting
# For fusion with matmul, can only support unidirectional broadcasting with matmul output
# as the first input
elementwise_two_input_ops = {
    "Add": ElementwiseOp(triton_template="{in0} + {in1}"),
    "Div": ElementwiseOp(triton_template="{in0} / {in1}"),
    "Mul": ElementwiseOp(triton_template="{in0} * {in1}"),
    "Pow": ElementwiseOp(triton_template="tl.pow({in0}, {in1})"),
    "Sub": ElementwiseOp(triton_template="{in0} - {in1}"),
}
