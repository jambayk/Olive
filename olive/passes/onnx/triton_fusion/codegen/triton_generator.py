# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Dict, List, Tuple

import olive.passes.onnx.triton_fusion.codegen.triton_templates as templates
from olive.passes.onnx.triton_fusion.codegen.ops import get_num_op_inputs, get_op_info
from olive.passes.onnx.triton_fusion.utils import TL_DTYPE_MAP, create_triton_kernel_name, join_params


def create_template_arg(op: str, op_idx: int, in_ptr: str, out_ptr: str) -> Dict:
    """Create the op related template arguments to use in the fused op template.

    Currently, we only support elementwise ops.
    """
    # number of inputs for this op
    # first input is always the output of the previous op
    num_inputs = get_num_op_inputs(op)
    op_info = get_op_info(op)
    # unique name for this op
    unique_op_name = f"{op}_{op_idx}".lower()

    # args to be generate the op code
    code_args = {"in0": in_ptr}
    # args to be used in the full template
    template_args = {"op_name": op.lower(), "ptr_param": None, "numel_param": None, "attr_params": [], "code": None}

    # create arg for second input if exists
    if num_inputs == 2:
        # the op needs to index and load the second input
        in1 = f"{unique_op_name}_in1"
        code_args["in1"] = in1
        code_args["in1_ptr"] = f"{in1}_ptr"
        code_args["in1_numel"] = f"{in1}_numel"
        template_args["ptr_param"] = f"{in1}_ptr"
        template_args["numel_param"] = f"{in1}_numel"

    # create unique temp var if needed
    for tmp_idx in range(op_info.num_temp_vars):
        tmp_ptr = f"{unique_op_name}_tmp{tmp_idx}"
        code_args[f"tmp{tmp_idx}"] = tmp_ptr

    # create args for attributes if any
    for attr_name, _ in op_info.attributes or []:
        attr_arg = f"{unique_op_name}_{attr_name}"
        code_args[attr_name] = attr_arg
        template_args["attr_params"].append(attr_arg)

    # create operator code
    operator_code = ""
    if isinstance(op_info.triton_template, str):
        # single line op where the output needs to be assigned
        operator_code += f"{out_ptr} = {op_info.triton_template.format(**code_args)}"
    elif isinstance(op_info.triton_template, list):
        # multi-line op where the last line needs to be assigned to the output
        for line in op_info.triton_templates:
            operator_code += f"{line.format(**code_args)}\n    "
        operator_code += f"{out_ptr} = {op_info.triton_template[-1].format(**code_args)}"

    # full code for this op
    full_code = f"# Op: {op}"
    if num_inputs == 1:
        full_code += f"\n    {operator_code}"
    else:
        code_args["op_code"] = operator_code
        full_code += templates.FUSED_OP_TWO_INPUT_TEMPLATE.format(**code_args)
    template_args["code"] = full_code

    return template_args


def create_kernel(base_op: str, fused_ops: List[str], dtype: str) -> Tuple[str, str]:
    """Create the kernel for the fused op.

    Returns the kernel name and the kernel code.
    """
    op_names = [base_op, *fused_ops]
    template = templates.MATMUL_TEMPLATE if base_op == "MatMul" else templates.ELEMENTWISE_TEMPLATE

    # create args for fused ops
    ptr_params = []
    numel_params = []
    attr_params = []
    codes = []
    for op_idx, op in enumerate(fused_ops if base_op == "MatMul" else op_names):
        template_args = create_template_arg(op, op_idx, "y", "y")
        if template_args["ptr_param"]:
            ptr_params.append(template_args["ptr_param"])
            numel_params.append(template_args["numel_param"])
        attr_params.extend(template_args["attr_params"] or [])
        codes.append(template_args["code"])
    kernel_name = create_triton_kernel_name(op_names, dtype)
    template_args = {
        "y_dtype": TL_DTYPE_MAP[dtype],
        "fused_ops_str": ", ".join(op_names),
        "kernel_name": kernel_name,
        "ptr_params": join_params(ptr_params),
        "numel_params": join_params(numel_params),
        "attr_params": join_params(attr_params),
        "fused_code": join_params(
            codes,
            joiner="\n\n    ",
            end="",
            default="# No fused op" if base_op == "MatMul" else "# This should not happen!",
        ),
    }

    # create full kernel
    return kernel_name, template.format(**template_args)
