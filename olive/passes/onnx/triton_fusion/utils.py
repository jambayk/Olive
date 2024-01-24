# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

TL_DTYPE_MAP = {
    "fp32": "tl.float32",
    "fp16": "tl.float16",
    "int32": "tl.int32",
    "int64": "tl.int64",
    "bool": "tl.bool",
    "bf16": "tl.bfloat16",
}


def get_env_path(var_name):
    if not os.environ.get(var_name):
        raise RuntimeError(f"{var_name} not set")

    return Path(os.environ[var_name])
