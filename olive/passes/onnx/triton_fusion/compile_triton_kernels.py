# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import triton

from olive.common.utils import run_subprocess

# ruff: noqa: N806


def compile_kernel(out_dir, signature, kernel_name, out_name, out_path, num_stages, num_warps, grid, kernel_path):
    compiler_path = Path(triton.tools.__path__[0]) / "compile.py"

    run_subprocess(
        f"python {compiler_path} -n {kernel_name} --signature {signature} --out-name {out_name} -o {out_path} -ns"
        f" {num_stages} -w {num_warps} -g {grid} {kernel_path}",
        check=True,
        cwd=out_dir,
    )


def compile_matmul_kernel(out_dir, dtype, BM, BN, BK, GM):
    sig = f"'*{dtype}, *{dtype}, *{dtype}, i32, i32, i32, {BM}, {BN}, {BK}, {GM}'"
    name = f"matmul_{dtype}"
    grid = f"'((M + {BM -1})/{BM}) * ((N + {BN - 1})/{BN}), 1, 1'"
    compile_kernel(
        out_dir=out_dir,
        signature=sig,
        kernel_name="matmul_kernel",
        out_name=name,
        out_path=name,
        num_stages=5,
        num_warps=2,
        grid=grid,
        kernel_path=Path(__file__).parent / "kernels.py",
    )


def link_matmul_kernels(out_dir, dtype):
    linker_path = Path(triton.tools.__path__[0]) / "link.py"

    # link all desired configs
    h_files = [str(file) for file in Path(out_dir).glob(f"matmul_{dtype}*.h")]
    run_subprocess(
        f"python {linker_path} {' '.join(h_files)} -o matmul_{dtype} -o matmul_{dtype}", check=True, cwd=out_dir
    )

    # need to add extern C to the header file to avoid name mangling
    # header is used in c++ code also
    extern_c_start = ["#ifdef __cplusplus\n", 'extern "C" {\n', "#endif\n", "\n"]
    extern_c_end = ["\n", "#ifdef __cplusplus\n", "}\n", "#endif\n"]

    lines = []
    with (out_dir / f"matmul_{dtype}.h").open() as f:
        lines = f.readlines()
    lines = lines[:2] + extern_c_start + lines[2:] + extern_c_end
    with (out_dir / f"matmul_{dtype}.h").open("w") as f:
        f.writelines(lines)


def create_triton_kernels(out_dir, dtype):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    BM, BN, BK, GM = 32, 64, 32, 8  # no
    compile_matmul_kernel(out_dir, dtype, BM, BN, BK, GM)
    link_matmul_kernels(out_dir, dtype)
