# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Dict, List, Tuple

from olive.passes.onnx.triton_fusion.codegen.ops import get_num_op_inputs, get_op_info
from olive.passes.onnx.triton_fusion.codegen.triton_templates import (
    ELEMENTWISE_TEMPLATE,
    FUSED_OP_TWO_INPUT_TEMPLATE,
    MATMUL_TEMPLATE,
)
from olive.passes.onnx.triton_fusion.utils import TL_DTYPE_MAP


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
    template_args = {"op_name": op.lower(), "ptr_arg": None, "numel_arg": None, "attr_args": [], "code": None}

    # create arg for second input if exists
    if num_inputs == 2:
        # the op needs to index and load the second input
        in1 = f"{unique_op_name}_in1"
        code_args["in1"] = in1
        code_args["in1_ptr"] = f"{in1}_ptr"
        code_args["in1_numel"] = f"{in1}_numel"
        template_args["ptr_arg"] = f"{in1}_ptr"
        template_args["numel_arg"] = f"{in1}_numel"

    # create unique temp var if needed
    for tmp_idx in range(op_info.num_temp_vars):
        tmp_ptr = f"{unique_op_name}_tmp{tmp_idx}"
        code_args[f"tmp{tmp_idx}"] = tmp_ptr

    # create args for attributes if any
    for attr_name, _ in op_info.attributes or []:
        attr_arg = f"{unique_op_name}_{attr_name}"
        code_args[attr_name] = attr_arg
        template_args["attr_args"].append(attr_arg)

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
        full_code += FUSED_OP_TWO_INPUT_TEMPLATE.format(**code_args)
    template_args["code"] = full_code

    return template_args


def create_kernel(base_op: str, fused_ops: List[str], dtype: str) -> Tuple[str, str]:
    """Create the kernel for the fused op.

    Returns the kernel name and the kernel code.
    """
    op_names = [base_op, *fused_ops]
    template = MATMUL_TEMPLATE if base_op == "MatMul" else ELEMENTWISE_TEMPLATE

    # create args for base op
    base_op_args = {"y_dtype": TL_DTYPE_MAP[dtype]}
    if base_op != "MatMul":
        template_args = create_template_arg(base_op, 0, "a", "y")
        base_op_args = {
            "b_ptr_arg": template_args["ptr_arg"] + "," if template_args["ptr_arg"] else "# Not used",
            "b_numel_arg": template_args["numel_arg"] + "," if template_args["numel_arg"] else "# Not used",
            "base_attr_args": ",\n    ".join(template_args["attr_args"]) + ","
            if template_args["attr_args"]
            else "# Not used",
            "base_code": template_args["code"],
        }

    # create args for fused ops
    fused_ptr_args = []
    fused_numel_args = []
    fused_attr_args = []
    fused_code = []
    for op_idx, op in enumerate(fused_ops):
        template_args = create_template_arg(op, op_idx + 1, "y", "y")
        if template_args["ptr_arg"]:
            fused_ptr_args.append(template_args["ptr_arg"])
            fused_numel_args.append(template_args["numel_arg"])
        fused_attr_args.extend(template_args["attr_args"] or [])
        fused_code.append(template_args["code"])
    template_args = {
        "fused_ops_str": ", ".join(op_names),
        "kernel_name": dtype + ("_" + "_".join([op.lower() for op in op_names]) if op_names else ""),
        "fused_ptr_args": ",\n    ".join(fused_ptr_args) + "," if fused_ptr_args else "# Not used",
        "fused_numel_args": ",\n    ".join(fused_numel_args) + "," if fused_numel_args else "# Not used",
        "fused_attr_args": ",\n    ".join(fused_attr_args) + "," if fused_attr_args else "# Not used",
        "fused_code": "\n\n    ".join(fused_code) if fused_code else "# No fused ops",
    }

    # create full kernel
    return f"triton_{template_args['kernel_name']}", template.format(**base_op_args, **template_args)
